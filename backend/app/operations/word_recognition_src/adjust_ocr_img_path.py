# in bash call: python backend/app/operations/word_recognition_src/adjust_ocr_img_path.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# adjust_ocr_img_path.py

import json
from pathlib import Path

def fix_input_paths(ocr_dir: Path, new_img_dir: Path):
    """
    Update the 'input_path' field in all OCR JSON files to point to the new image directory.

    Args:
        ocr_dir (Path): Directory containing OCR result JSON files.
        new_img_dir (Path): New relative or absolute directory for image paths.
    """
    ocr_dir = ocr_dir.resolve()
    new_img_dir = new_img_dir.as_posix()  # always use forward slashes

    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")

    json_files = list(ocr_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {ocr_dir}")
        return

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_filename = Path(data["input_path"]).name
        data["input_path"] = f"{new_img_dir}/{image_filename}"

        with json_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated {len(json_files)} files in '{ocr_dir}' with image path prefix '{new_img_dir}/'")


def main():
    # EDIT THESE PATHS AS NEEDED
    # Accepts absolute or relative paths
    ocr_dir = Path("backend/app/operations/word_recognition_src/data/testing/page_19/character_recognition/output")
    new_img_dir = Path("testing/page_19/isolated_chars")  # e.g. "input" or any relative folder
    '''
    inference.py expects paths that start at "backend/app/operations/word_recognition_src/data/"
    '''

    fix_input_paths(ocr_dir, new_img_dir)

if __name__ == "__main__":
    main()