import os
import json
from pathlib import Path
from datetime import datetime

def jsons_to_txt(
    json_source: str,
    output_txt: str = None,
    keep_keys: list = None
) -> str:
    """
    Converts JSON file(s) to a single text file, one JSON record per line.
    Null-valued fields are dropped. Optionally only a subset of keys is written.

    Args:
        json_source (str): Path to a JSON file or directory of JSON files.
        output_txt (str, optional): Desired output text file path. If omitted,
            a timestamped file is created in cwd.
        keep_keys (list, optional): If provided, only these keys (plus "File Name")
            will be included in each output record.

    Returns:
        str: Path to the saved text file, or None on error.
    """
    src = Path(json_source)
    # Gather JSON files
    if src.is_file() and src.suffix.lower() == ".json":
        files = [src]
    elif src.is_dir():
        files = list(src.glob("*.json"))
    else:
        print(f"Error: '{json_source}' is not a valid JSON file or directory.")
        return None

    if not files:
        print(f"No JSON files found in: {json_source}")
        return None

    rows = []
    for filepath in files:
        try:
            raw = filepath.read_text(encoding='utf-8')
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Skipping {filepath.name}: parse error ({e})")
            continue

        # Flatten one level: dicts or list-of-dicts
        def extract(obj, base=None):
            base = base or {}
            if isinstance(obj, dict):
                return [{**base, **obj}]
            if isinstance(obj, list):
                out = []
                for item in obj:
                    if isinstance(item, dict):
                        out.extend(extract(item, base))
                return out
            return []

        records = extract(data, base={"File Name": filepath.name})
        for rec in records:
            # Drop keys with None values
            cleaned = {k: v for k, v in rec.items() if v is not None}
            # If keep_keys is set, filter to only those
            if keep_keys:
                allowed = set(["File Name", *keep_keys])
                cleaned = {k: cleaned[k] for k in cleaned if k in allowed}
            # Skip completely empty records
            if len(cleaned) > 1 or ("File Name" in cleaned and len(cleaned) == 1):
                rows.append(cleaned)

    if not rows:
        print("No data extracted from any JSON files.")
        return None

    # Determine output path
    if output_txt:
        out_path = Path(output_txt)
        if out_path.suffix.lower() != ".txt":
            out_path = out_path.with_suffix(".txt")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"output_{timestamp}.txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write each record as a JSON line
    try:
        with out_path.open('w', encoding='utf-8') as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        print(f"Text file saved to: {out_path}")
        return str(out_path)
    except Exception as e:
        print(f"Failed to save text file: {e}")
        return None



    
