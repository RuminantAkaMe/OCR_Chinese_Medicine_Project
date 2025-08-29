# in bash call: python backend/app/operations/word_recognition_src/fix_img_paths.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# continuation of convert_trainingset.py, put the files in the rootfolder where this script is called from
# fix_img_paths.py
import json
from pathlib import Path

def fix_paths(in_file="page19_train.jsonl", out_file="page19_train_fixed.jsonl", zero_pad=3):
    """
    Fix image paths inside a JSONL training dataset.

    Each record in the input JSONL contains a list of "input" items, where
    each item has an "id" and an "img" path like "data/input/002.png".

    This function rewrites the "img" filename so that it matches the value
    of the "id". For example:
        {"id": 26, "img": "data/input/002.png"}
    becomes:
        {"id": 26, "img": "data/input/026.png"}

    Args:
        in_file (str): Path to the input JSONL file.
        out_file (str): Path to the output JSONL file with corrected paths.
        zero_pad (int): Number of digits to pad the ID with (default: 3).
                        Example: id=7 → "007.png"
    """
    in_p = Path(in_file)
    out_p = Path(out_file)

    count = 0
    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            for item in rec.get("input", []):
                try:
                    _id = int(item["id"])
                except Exception:
                    continue  # If no valid integer ID, leave unchanged

                # Determine file extension from existing path (default: .png)
                old = item.get("img", "data/input/000.png")
                suffix = Path(old).suffix or ".png"

                # Replace filename with zero-padded ID
                new_name = f"{_id:0{zero_pad}d}{suffix}"
                item["img"] = "data/input/" + new_name

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} lines to {out_p}")


if __name__ == "__main__":
    # Run directly: fixes image paths in the JSONL file
    fix_paths()

