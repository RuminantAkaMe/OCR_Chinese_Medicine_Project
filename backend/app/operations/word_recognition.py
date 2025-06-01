# word_recognition.py

import json
from pathlib import Path
from app.operations.word_recognition_src.image_to_feature_vector import image_to_feature_vector
from app.operations.word_recognition_src.model.llm_prediction import query_model

def run() -> str:
    """
    Executes LLM-based word recognition.
    Constructs a structured prompt using OCR and image embeddings.
    Returns the output path of the JSON result.
    """

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "word_recognition_src" / "data"

    sequence_path = DATA_DIR / "sequence.json"
    output_path = DATA_DIR / "output.json"

    # Load input sequence (OCR + image paths)
    with sequence_path.open("r", encoding="utf-8") as f:
        sequence = json.load(f)

    # Construct prompt with OCR and truncated image embeddings
    lines = []
    for token in sequence:
        ocr = token["ocr"]
        img_path = (sequence_path.parent / token["img"]).resolve()  # Resolve full image path
        emb = image_to_feature_vector(img_path)                     # Convert image to vector
        emb_str = ",".join(f"{v:.4f}" for v in emb[:16])            # Use first 16 dims only
        lines.append(f"{token['id']}: {ocr} <IMG:{emb_str}>")

    prompt = "\n".join(lines)

    # Run the LLM prediction on the constructed prompt
    result = query_model(prompt)

    # Save the result to output.json
    with output_path.open("w", encoding="utf-8") as f:
        f.write(result)

    return str(output_path)




