# word_recognition.py

from pathlib import Path
import json
from app.operations.word_recognition_src.model.llm_prediction import query_model
from app.operations.word_recognition_src.json_to_img import render_raw_json_to_image

def run() -> str:
    """
    Executes the word recognition pipeline using SmolVLM2.

    This function loads a character sequence (with image references and OCR text),
    passes it to the large language model for inference, and stores the raw result.

    Returns:
        str: Path to the JSON file containing the model's response.
    """
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "word_recognition_src" / "data"
    sequence_path = DATA_DIR / "sequence.json"
    output_path = DATA_DIR / "output.json"

    # Perform inference on the complete input sequence using the language model
    result = query_model(sequence_path)

    # Save the raw model output to a file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"response": result}, f, ensure_ascii=False, indent=2)

    return render_raw_json_to_image(str(output_path))








