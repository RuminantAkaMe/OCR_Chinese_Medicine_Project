# in bash call: python backend/app/operations/word_recognition_src/fix_paths_unique.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# changes data image paths to make it compatible
# fix_paths_unique.py

import json
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fix_file(jsonl_path: Path):
    """
    Simple script changes the image paths in the *_train.jsonl files to make them complatible my having images stored in pagewise folders.
    """
    page_name = jsonl_path.parent.name   # z.B. "page15"
    new_records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for item in rec.get("input", []):
                filename = Path(item["img"]).name
                # new path
                new_path = f"data/{page_name}/images/{filename}"
                item["img"] = new_path
            new_records.append(rec)

    out_path = OUT_DIR / f"{page_name}_train.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in new_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅ {jsonl_path} → {out_path}")

def main():
    for jsonl_path in DATA_DIR.glob("page*/**/*_train.jsonl"):
        fix_file(jsonl_path)

if __name__ == "__main__":
    main()

