# in bash call: python backend/app/operations/word_recognition_src/remove_confidence.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# 4. Script in dataset adjustment process
# remove_confidence.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "noconf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_file(jsonl_path: Path):
    """
    Simple script that removed the confidene value from the dataset in the output as it is computed by logprobs and not part of training data.
    """
    page_name = jsonl_path.parent.name
    out_path = OUT_DIR / f"{page_name}_train.jsonl"

    with jsonl_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("output", {})

            # 1) output.confidence
            if isinstance(out, dict):
                out.pop("confidence", None)

                # 2) output.best.confidence
                best = out.get("best")
                if isinstance(best, dict):
                    best.pop("confidence", None)

                # 3) output.candidates[*].confidence
                cands = out.get("candidates")
                if isinstance(cands, list):
                    for c in cands:
                        if isinstance(c, dict):
                            c.pop("confidence", None)

            rec["output"] = out
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {jsonl_path} → {out_path}")

def main():
    for jsonl_path in DATA_DIR.glob("page*/**/*_train.jsonl"):
        process_file(jsonl_path)

if __name__ == "__main__":
    main()

