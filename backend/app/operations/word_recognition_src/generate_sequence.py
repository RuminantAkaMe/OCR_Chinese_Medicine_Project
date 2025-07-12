# in bash call: python backend/app/operations/word_recognition_src/generate_sequence.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# generate_sequence.py

from pathlib import Path
import json
from tqdm import tqdm

def create_sequence_json(
    ocr_dir: Path,
    sequence_path: Path
):
    """
    Create a sequence.json file from OCR metadata, keeping the original input_path.
    No images are copied or renamed.

    Args:
        ocr_dir (Path): Directory containing JSON files with OCR results.
        sequence_path (Path): Output path for the generated sequence.json.
    """
    sequence = []

    ocr_files = sorted(ocr_dir.glob("*.json"))

    for idx, json_file in enumerate(tqdm(ocr_files, desc="Building sequence (keep original paths)")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sequence.append({
            "id": idx,
            "ocr": data["rec_text"],
            "ocr_confidence": round(data["rec_score"], 4),
            "img": data["input_path"].replace("\\", "/")  # for consistent paths
        })

    with sequence_path.open("w", encoding="utf-8") as f:
        json.dump(sequence, f, ensure_ascii=False, indent=4)

    print(f"Saved {sequence_path} with {len(sequence)} entries.")


def main():
    script_dir = Path(__file__).resolve().parent

    create_sequence_json(
        ocr_dir=script_dir / "data" / "testing" / "page_19" / "character_recognition" / "output",
        sequence_path=script_dir / "data" / "sequence.json"
    )

if __name__ == "__main__":
    main()


