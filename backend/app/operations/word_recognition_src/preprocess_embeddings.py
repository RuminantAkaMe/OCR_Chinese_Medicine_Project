# preprocess_embeddings.py

import json
from pathlib import Path
from processor.image_to_vector import image_to_feature_vector

# Paths to input sequence and output training data
INPUT_PATH = Path("app/operations/word_recognition_src/data/sequence.json")
OUTPUT_PATH = Path("app/operations/word_recognition_src/data/train.jsonl")

# Load token sequence with OCR and image paths
with INPUT_PATH.open("r", encoding="utf-8") as f:
    sequence = json.load(f)

# Generate input prompt with OCR + image embedding (truncated) as tokens
lines = []
for token in sequence:
    ocr = token["ocr"]                             # Extract OCR text
    img_path = Path(token["img"])                  # Get image path
    emb = image_to_feature_vector(img_path)        # Extract image embedding
    emb_str = ",".join(f"{v:.4f}" for v in emb[:16])  # Convert first 16 values to string
    lines.append(f"{token['id']}: {ocr} <IMG:{emb_str}>")

# Combine all lines into the final prompt
input_prompt = "\n".join(lines)

# Dummy output placeholder — to be replaced with model-generated output later
output_json = [
    {
        "span": [0, 1, 2],
        "candidates": [
            {"word": "牛黃散", "confidence": 0.91},
            {"word": "牛用散", "confidence": 0.35}
        ]
    }
]

# Save the input-output pair in JSON Lines format for training purposes
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump({
        "input": input_prompt,
        "output": json.dumps(output_json, ensure_ascii=False)
    }, f, ensure_ascii=False)
    f.write("\n")

