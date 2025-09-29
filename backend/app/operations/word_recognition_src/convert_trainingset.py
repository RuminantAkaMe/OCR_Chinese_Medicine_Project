# in bash call: python backend/app/operations/word_recognition_src/convert_trainingset.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# put the dataset.json in the rootfolder where this script is called from
# 1. Script in dataset adjustment process
# convert_trainingset.py
import json, re
from pathlib import Path
from typing import List, Dict, Any
import random

def extract_candidates_ordered(raw: str) -> List[Dict[str, Any]]:
    """
    Extract all (text, confidence) pairs from the 'candidates' block in order.

    This function is robust against malformed JSON structures where multiple
    "text" and "confidence" keys are written inside a single dictionary.
    It uses regex to recover them in sequential order.

    Args:
        raw (str): Raw JSON file contents as string.

    Returns:
        List[Dict[str, Any]]: A list of candidate dictionaries with fields:
                              {"text": <string>, "confidence": <float>}
    """
    m = re.search(r'"candidates"\s*:\s*\[(.*?)\]', raw, flags=re.S)
    if not m:
        return []
    block = m.group(1)
    text_hits = list(re.finditer(r'"text"\s*:\s*"([^"]+)"', block))
    conf_pat = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')
    out = []
    for i, th in enumerate(text_hits):
        text_val = th.group(1)
        start = th.end()
        end = text_hits[i+1].start() if i+1 < len(text_hits) else len(block)
        slice_ = block[start:end]
        cm = conf_pat.search(slice_)
        conf_val = float(cm.group(1)) if cm else 1.0
        out.append({"text": text_val, "confidence": conf_val})
    return out


def convert_dataset(input_file: str = "dataset.json", output_file: str = "train.jsonl"):
    """
    Convert a dataset.json file into a train.jsonl file suitable for training.

    Each (span, candidate) pair becomes one JSON line in the output.
    The 'input' section contains only the characters belonging to the span,
    preserving their original IDs. The 'output' section contains exactly one
    candidate word with its confidence and the corresponding global span.

    If no OCR confidence is provided for a character, a random value between
    0.75 and 0.97 (rounded to 2 decimals) will be assigned.

    Args:
        input_file (str): Path to dataset.json file.
        output_file (str): Path to output train.jsonl file.
    """
    in_path = Path(input_file)
    out_path = Path(output_file)

    raw = in_path.read_text(encoding="utf-8")
    data = json.loads(raw)  # standard JSON parsing; candidates are extracted separately

    seq = data.get("input", [])
    spans = data.get("output", {}).get("spans", [])
    candidates = extract_candidates_ordered(raw)

    if not seq or not spans or not candidates:
        raise ValueError("Input, spans or candidates are missing/empty.")

    n = min(len(spans), len(candidates))
    if len(spans) != len(candidates):
        print(f"Warning: spans ({len(spans)}) != candidates ({len(candidates)}). "
              f"Processing only the first {n} pairs in order.")

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(n):
            start, end = spans[i]
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(seq)):
                print(f"⚠️ Skipping invalid span #{i}: {spans[i]}")
                continue

            # Subsequence extraction – IDs remain ORIGINAL
            sub = seq[start:end+1]
            new_input = [{
                "id": item["id"],   # preserve original ID
                "img": item["img"],
                "ocr": item["ocr"],
                "ocr_confidence": float(item.get(
                    "ocr_confidence",
                    round(random.uniform(0.75, 0.97), 2)  # fallback random confidence
                )),
            } for item in sub]

            cand = candidates[i]
            record = {
                "input": new_input,
                "output": {
                    "candidates": [{
                        "text": cand["text"],
                        "confidence": float(cand.get("confidence", 1.0))
                    }],
                    "spans": [spans[i]],  # keep original global span
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. {written} training lines written to {out_path}")


if __name__ == "__main__":
    # Run from project root:
    #   python backend/app/operations/word_recognition_src/convert_trainingset.py
    convert_dataset("dataset.json", "train.jsonl")
