# in bash call: python backend/app/operations/word_recognition_src/make_spans_relative.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# put the files in the rootfolder where this script is called from
# 5. Script in dataset adjustment process
# make_spans_relative.py
import json
from pathlib import Path

def make_spans_relative(in_file="page6_train.jsonl", out_file="page6_train_rel.jsonl"):
    """
    Adjust spans in a JSONL training dataset so that they are relative
    to the first input ID in each record.

    Args:
        in_file (str): Input JSONL file path.
        out_file (str): Output JSONL file path with adjusted spans.
    """
    in_p = Path(in_file)
    out_p = Path(out_file)

    changed, total = 0, 0
    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                fout.write(line + "\n")
                continue

            inp = rec.get("input", [])
            if not inp or not isinstance(inp, list):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # erste id vom input
            try:
                first_id = int(inp[0].get("id"))
            except Exception:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            out = rec.get("output")
            if isinstance(out, dict) and "spans" in out:
                spans = out.get("spans", [])
                new_spans = []
                for s in spans:
                    if isinstance(s, (list, tuple)) and len(s) == 2:
                        try:
                            a, b = int(s[0]) - first_id, int(s[1]) - first_id
                            new_spans.append([a, b])
                        except Exception:
                            new_spans.append(s)
                    else:
                        new_spans.append(s)
                if new_spans != spans:
                    out["spans"] = new_spans
                    changed += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ {in_p} → {out_p}  (geänderte Zeilen: {changed}/{total})")


if __name__ == "__main__":
    # Run directly: adjust spans in the JSONL file
    make_spans_relative()
