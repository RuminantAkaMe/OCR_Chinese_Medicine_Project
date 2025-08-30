# in bash call: python backend/app/operations/word_recognition_src/model/eval_llava_metrics.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# eval_llava_metrics.py
"""
Evaluation script aligned with the papers definitions.

What this script does
---------------------
1) Loads the fold's validation JSONL from ./data/train.jsonl (as prepared by kfold_train.py).
2) Builds inputs (PIL images + OCR prompt) exactly like training.
3) Runs generation and parses the models JSON output.
4) Builds a ranked candidate list per example:
   - Primary path: use candidates from the model's JSON if they include per-candidate 'confidence'.
   - Fallback path: if no usable candidates, generate up to K candidates (do_sample) and compute
     each candidates sequence confidence from token log-probabilities; rank by that.
5) Computes metrics:
   a) Top-k Exact Match (EM@k): ground-truth ∈ top-k candidates (ranked by confidence).
   b) Sequence-level Accuracy (SeqAcc@k): same as EM@k if there is exactly one target per sequence
      (your current data); for multiple targets it would require all targets to appear within top-k.
   c) Span F1 (exact span-pair match, macro-averaged across sequences).
   d) Expected Calibration Error (ECE): binned ECE over confidence vs correctness.
   e) Brier Score: mean squared error of confidence vs correctness.

Formulas implemented
--------------------
Let N be the number of sequences (examples). For sequence i:
- y_i : ground-truth word (string)
- Ĉ_i = [(c_{i1}, p_{i1}), ..., (c_{ik}, p_{ik}), ...] : ranked candidates with confidences p (descending)
- 1[·] : indicator function

(1) Top-k Exact Match (EM@k)
    EM@k = (1/N) * sum_{i=1..N} 1[ y_i ∈ TopK(Ĉ_i) ]

(2) Sequence-level Accuracy (SeqAcc@k)
    If each sequence has exactly one target (your case), SeqAcc@k ≡ EM@k.
    (For multiple targets T_i, one would require all targets ∈ TopK(Ĉ_i).)

(3) Span F1 (exact span-pair match)
    For sequence i, let S_i be the set of gold spans and P_i the set of predicted spans
    (each span as an exact (start, end) pair).
    precision_i = |S_i ∩ P_i| / |P_i|   (0 if |P_i|=0)
    recall_i    = |S_i ∩ P_i| / |S_i|   (0 if |S_i|=0)
    F1_i        = 2 * precision_i * recall_i / (precision_i + recall_i)  (0 if denom=0)
    SpanF1 = (1/N) * sum_{i=1..N} F1_i

(4) Expected Calibration Error (ECE; M bins)
    Partition predictions into bins B_m by confidence. For bin m:
      acc(B_m)  = mean(correctness in B_m)
      conf(B_m) = mean(confidence in B_m)
    ECE = sum_{m=1..M} (|B_m|/N) * | acc(B_m) - conf(B_m) |

(5) Brier Score
    With correctness ŷ_i ∈ {0,1} and predicted confidence p_i ∈ [0,1]:
    Brier = (1/N) * sum_{i=1..N} (p_i - ŷ_i)^2

Sequence confidence from token log-probabilities
------------------------------------------------
Given generated token ids z_{1:T} and per-step pre-softmax distributions:
    log p_seq = (1/T) * sum_{t=1..T} log softmax(scores_t)[ z_t ]
    conf_seq  = exp( log p_seq )   # geometric mean of per-token probabilities

Notes
-----
- Primary candidate ranking uses confidences from the models JSON if present.
- Fallback uses multiple generations (do_sample) to approximate a Top-k list; each candidate
  receives a confidence computed from token log-probs as above. Duplicates are de-duplicated,
  keeping the highest confidence.
- Paths: images are resolved like in training via Path(__file__).parent.parent / token["img"].
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.metrics import brier_score_loss

import torch
from torch.nn.functional import log_softmax
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel
from PIL import Image

# -------------------- Config --------------------
# Resolve paths in a cross-platform way (Windows/Linux) using pathlib.
# BASE_DIR is the project base for word_recognition_src (…/word_recognition_src).
BASE_DIR = Path(__file__).resolve().parent.parent
# If called via kfold, CWD is .../checkpoints_cv/foldX/
# -> write into .../word_recognition_src/data/eval_out/foldX
# Else (standalone), write into .../word_recognition_src/data/eval_out
cwd = Path.cwd()
if (cwd / "checkpoints").exists() and cwd.name.startswith("fold"):
    OUTPUT_DIR = BASE_DIR / "data" / "eval_out" / cwd.name
else:
    OUTPUT_DIR = BASE_DIR / "data" / "eval_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# DATA_PATH: prefer the script-relative path (standalone),
# but also works when launched from kfold (same relative layout).
DATA_PATH = BASE_DIR / "data" / "train.jsonl"

# MODEL_DIR: prefer the fold-local ./checkpoints (when run via kfold, CWD=foldX),
# otherwise fall back to script-relative …/word_recognition_src/checkpoints.
_fold_checkpoints = Path("./checkpoints")  # kfold case (relative to current working dir)
if (_fold_checkpoints / "adapter_config.json").exists() or any(_fold_checkpoints.glob("checkpoint-*/adapter_config.json")):
    MODEL_DIR = _fold_checkpoints
else:
    MODEL_DIR = BASE_DIR / "checkpoints"   # standalone case

# NOTE:
# - Always pass MODEL_DIR as a string to PeftModel.from_pretrained(...): str(MODEL_DIR)
# - pathlib handles Windows/Linux path separators internally; no manual os.path needed.
# For HPC Usage:
# MODEL_BASE = str(Path(__file__).resolve().parent.parent / "model" / "llava-onevision-qwen2-0.5b-ov-hf")
MODEL_BASE = "E:/Software-Projekte/Llava/llava-onevision-qwen2-0.5b-ov-hf"
TOP_K      = [1, 5]
ECE_BINS   = 10
FALLBACK_MAX_K = max(TOP_K)  # number of sequences to sample if no candidates are provided

# Generation params for fallback Top-k (kept conservative)
FALLBACK_GEN = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=256,
    return_dict_in_generate=True,
    output_scores=True,
)

# -------------------- Model/processor --------------------
processor = AutoProcessor.from_pretrained(MODEL_BASE)
base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_BASE, device_map="auto", torch_dtype=torch.float32
)

def find_adapter(root=Path("./checkpoints")):
    cands = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for p in reversed(cands):
        if (p / "adapter_config.json").exists():
            return p
    if (root / "adapter_config.json").exists():
        return root
    return None

adapter = find_adapter()
if adapter:
    model = PeftModel.from_pretrained(base_model, str(adapter))
else:
    print("No LoRA adapter found; evaluating base model only.")
    model = base_model

model.eval()


# -------------------- IO helpers --------------------
def load_data(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_inputs(example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Builds inputs exactly like training: PIL images + OCR prompt via chat template."""
    imgs, ocrs = [], []
    for tok in example["input"]:
        img_path = Path(__file__).parent.parent / tok["img"]
        imgs.append(Image.open(img_path).convert("RGB"))
        ocrs.append(tok["ocr"])
    prompt = " ".join(ocrs)
    messages = [{
        "role": "user",
        "content": ([{"type": "image", "image": im} for im in imgs] +
                    [{"type": "text", "text": f"请根据这些图像（{prompt}）组成一个词语，并给出 JSON 格式的候选词及其范围。"}])
    }]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    return enc


def parse_json_safe(s):
    try:
        data = json.loads(s)
        if isinstance(data, list):
            # normalize to object shape
            return {"candidates": data, "best": (data[0] if data else None)}
        return data if isinstance(data, dict) else {"candidates": [], "best": None}
    except:
        return {"candidates": [], "best": None}


# -------------------- Confidence / candidates --------------------
def seq_confidence_from_scores(output, prompt_len: int, seq_index: int = 0) -> float:
    """
    Compute sequence confidence for a specific returned sequence:
    geometric mean of token probabilities for the generated segment.
    """
    seq_ids = output.sequences[seq_index][prompt_len:]
    scores = output.scores
    # For each time step t, scores[t] is logits over vocab for *all* sequences in the batch.
    # With batch size 1 and num_return_sequences=1, indexing is straightforward.
    # For fallback multi-generation we run generation per candidate to keep mapping trivial.
    lp = []
    for t, tok_id in enumerate(seq_ids):
        dist = scores[t]  # shape [vocab]
        if dist.dim() == 2:        # [batch(=1), vocab] → [vocab]
            dist = dist[0]
        lp.append(log_softmax(dist, dim=-1)[tok_id].item())
    return float(np.exp(np.mean(lp))) if lp else 0.0


def extract_best_and_spans(pred: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int]]]:
    """Extracts best.text and best.span from model JSON."""
    best = pred.get("best") or {}
    text = best.get("text", "") if isinstance(best, dict) else ""
    span = best.get("span", []) if isinstance(best, dict) else []
    # ensure span is list of (start, end) pairs
    spans = []
    for s in span:
        if isinstance(s, (list, tuple)) and len(s) == 2:
            spans.append((int(s[0]), int(s[1])))
    return text, spans


def candidates_from_json(pred: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    If model outputs candidates with confidences, return [(text, confidence)] sorted desc.
    """
    cands = pred.get("candidates")
    if not isinstance(cands, list) or not cands:
        return []
    results = []
    for c in cands:
        if isinstance(c, dict):
            t = c.get("text", "")
            p = c.get("confidence", None)
            if isinstance(t, str) and isinstance(p, (float, int)):
                results.append((t, float(p)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def fallback_candidates(enc: Dict[str, torch.Tensor], k: int) -> List[Tuple[str, float]]:
    """
    Generate up to k candidates by repeated stochastic decoding.
    Each candidate gets a confidence from token log-probs; duplicates are de-duplicated.
    """
    results: Dict[str, float] = {}
    for i in range(k):
        # Run per-candidate generation to keep score mapping simple.
        # (This is slower than num_return_sequences, but robust.)
        g = {k_: v_.to(model.device) for k_, v_ in enc.items()}
        with torch.no_grad():
            out = model.generate(**g, **FALLBACK_GEN)
        prompt_len = g["input_ids"].shape[-1]
        gen_text = processor.tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
        pred = parse_json_safe(gen_text)
        best_text, _ = extract_best_and_spans(pred)
        text = best_text if best_text else gen_text.strip()
        conf = seq_confidence_from_scores(out, prompt_len, seq_index=0)
        if text not in results or conf > results[text]:
            results[text] = conf
    # sort by confidence desc
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


# -------------------- Metrics --------------------
def topk_em(y_true: List[str], ranked_cands: List[List[Tuple[str, float]]], k: int) -> float:
    hits = 0
    for t, cand_list in zip(y_true, ranked_cands):
        texts = [c[0] for c in cand_list[:min(k, len(cand_list))]]
        hits += int(t in texts)
    return hits / max(1, len(y_true))


def seq_acc_k(y_true: List[str], ranked_cands: List[List[Tuple[str, float]]], k: int) -> float:
    # With one target per sequence, SeqAcc@k == EM@k.
    return topk_em(y_true, ranked_cands, k)


def span_f1_macro(true_spans: List[List[Tuple[int, int]]],
                  pred_spans: List[List[Tuple[int, int]]]) -> float:
    f1s = []
    for ts, ps in zip(true_spans, pred_spans):
        S = set(map(tuple, ts))
        P = set(map(tuple, ps))
        tp = len(S & P)
        fp = len(P - S)
        fn = len(S - P)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s) if f1s else 0.0)


def ece_binned(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """Binned ECE as in the paper."""
    assert len(confidences) == len(correctness)
    N = len(confidences)
    if N == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        left, right = bins[b], bins[b + 1]
        # include left edge; last bin includes right edge
        if b < n_bins - 1:
            idx = (confidences >= left) & (confidences < right)
        else:
            idx = (confidences >= left) & (confidences <= right)
        count = int(idx.sum())
        if count == 0:
            continue
        acc_b = float(correctness[idx].mean())
        conf_b = float(confidences[idx].mean())
        ece += (count / N) * abs(acc_b - conf_b)
    return float(ece)


# -------------------- Main evaluation --------------------
def evaluate():
    data = load_data(DATA_PATH)

    y_true: List[str] = []
    ranked_cands_per_ex: List[List[Tuple[str, float]]] = []
    best_preds: List[str] = []
    best_confidences: List[float] = []
    true_spans_all: List[List[Tuple[int, int]]] = []
    pred_spans_all: List[List[Tuple[int, int]]] = []

    for ex in tqdm(data, desc="Evaluating"):
        enc = build_inputs(ex)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        # First try a single deterministic-ish decode (greedy) to parse JSON and "best"
        with torch.no_grad():
            out1 = model.generate(
                **enc,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )
        prompt_len = enc["input_ids"].shape[-1]
        gen_text1 = processor.tokenizer.decode(out1.sequences[0][prompt_len:], skip_special_tokens=True)
        pred1 = parse_json_safe(gen_text1)

        # Candidate list:
        cand_list = candidates_from_json(pred1)

        # If no usable candidates from JSON, fallback to k stochastic samples
        if not cand_list:
            cand_list = fallback_candidates(enc, FALLBACK_MAX_K)

        # Best prediction string and its confidence
        if cand_list:
            best_text, best_conf = cand_list[0]
        else:
            # degenerate fallback: use single decode text + its confidence
            best_text, _ = extract_best_and_spans(pred1)
            if not best_text:
                best_text = gen_text1.strip()
            best_conf = seq_confidence_from_scores(out1, prompt_len, seq_index=0)
            cand_list = [(best_text, best_conf)]

        # Spans (from the single decode's "best"; span scoring is on best)
        best = pred1.get("best") or {}
        pred_spans = best.get("span") or pred1.get("spans") or []
        pred_spans = [(int(s[0]), int(s[1])) for s in pred_spans
                    if isinstance(s, (list, tuple)) and len(s) == 2]

        # --- Ground truth (fixed schema) ---
        gold = ex["output"]  # dict
        assert isinstance(gold, dict) and "candidates" in gold and gold["candidates"], \
            "Expected output.candidates to be a non-empty list"
        true_word = str(gold["candidates"][0].get("text", ""))
        true_spans = [(int(s[0]), int(s[1])) for s in gold.get("spans", [])
                    if isinstance(s, (list, tuple)) and len(s) == 2]

        # Accumulate
        y_true.append(true_word)
        ranked_cands_per_ex.append(cand_list)
        best_preds.append(best_text)
        best_confidences.append(best_conf)
        true_spans_all.append(true_spans)
        pred_spans_all.append(pred_spans)

    # ---- Metrics ----
    print("\n=== Evaluation Results (Paper-aligned) ===")
    for k in TOP_K:
        emk = topk_em(y_true, ranked_cands_per_ex, k)
        print(f"Top-{k} Exact Match (EM@{k}): {emk:.4f}")

    for k in TOP_K:
        seqk = seq_acc_k(y_true, ranked_cands_per_ex, k)
        print(f"Sequence-level Accuracy (SeqAcc@{k}): {seqk:.4f}")

    span_f1 = span_f1_macro(true_spans_all, pred_spans_all)
    print(f"Span F1 (exact span-pair, macro): {span_f1:.4f}")

    correctness = np.array([int(t == p) for t, p in zip(y_true, best_preds)], dtype=float)
    confidences = np.clip(np.array(best_confidences, dtype=float), 0.0, 1.0)
    ece = ece_binned(confidences, correctness, n_bins=ECE_BINS)
    print(f"Expected Calibration Error (ECE, {ECE_BINS} bins): {ece:.4f}")

    brier = brier_score_loss(correctness, confidences)
    print(f"Brier Score: {brier:.4f}")

        # ---------- Persist results to files ----------
    # Recompute EM@k/SeqAcc@k values so we can store them
    emk_vals = {f"em@{k}": topk_em(y_true, ranked_cands_per_ex, k) for k in TOP_K}
    seqk_vals = {f"seqacc@{k}": seq_acc_k(y_true, ranked_cands_per_ex, k) for k in TOP_K}

    # Confidence stats
    mean_conf = float(confidences.mean()) if len(confidences) else 0.0
    mean_conf_correct = float(confidences[correctness == 1].mean()) if (correctness == 1).any() else None
    mean_conf_wrong   = float(confidences[correctness == 0].mean()) if (correctness == 0).any() else None

    # 1) summary.json
    summary = {
        "num_samples": len(y_true),
        "topk": {**emk_vals, **seqk_vals},
        "span_f1_macro": float(span_f1),
        "ece_bins": int(ECE_BINS),
        "ece": float(ece),
        "brier": float(brier),
        "mean_conf": mean_conf,
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_wrong": mean_conf_wrong,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) ece_bins.csv
    import csv
    bins = np.linspace(0.0, 1.0, ECE_BINS + 1)
    with (OUTPUT_DIR / "ece_bins.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "count", "acc", "conf"])
        for b in range(ECE_BINS):
            left, right = bins[b], bins[b + 1]
            idx = (confidences >= left) & (confidences < (right if b < ECE_BINS - 1 else right + 1e-9))
            cnt = int(idx.sum())
            if cnt == 0:
                continue
            acc_b = float(correctness[idx].mean())
            conf_b = float(confidences[idx].mean())
            w.writerow([left, right, cnt, acc_b, conf_b])

    # Helper for per-sample span F1 (exact span-pair match)
    def span_f1_one(ts, ps):
        S, P = set(map(tuple, ts)), set(map(tuple, ps))
        tp = len(S & P); fp = len(P - S); fn = len(S - P)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # 3) per-sample details → eval_details.csv
    with (OUTPUT_DIR / "eval_details.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true", "pred", "conf", "hit@1", "hit@5", "span_f1", "num_cands"])
        for i, (t, cands, pred, conf, ts, ps) in enumerate(
            zip(y_true, ranked_cands_per_ex, best_preds, best_confidences, true_spans_all, pred_spans_all)
        ):
            hit1 = int(t == pred)
            top5 = [x[0] for x in cands[:5]]
            hit5 = int(t in top5)
            w.writerow([i, t, pred, float(conf), hit1, hit5, float(span_f1_one(ts, ps)), len(cands)])

    print(f"\nSaved files in: {OUTPUT_DIR.resolve()}")



if __name__ == "__main__":
    evaluate()
