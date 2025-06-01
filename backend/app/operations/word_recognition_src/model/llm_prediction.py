# model/llm_wrapper.py

from .mistral_model import generate_json

# Build a structured prompt from a sequence of OCR-annotated tokens.
# Each token is expected to be a dictionary with at least 'id' and 'ocr' keys.
# This prompt is designed for the LLM to identify meaningful multi-character words.
def build_prompt(sequence: list[dict]) -> str:
    # Format each token as "id: ocr" (e.g., "0: 牛")
    lines = [f"{token['id']}: {token['ocr']}" for token in sequence]
    
    # Join all lines into a single string separated by newlines
    joined_sequence = "\n".join(lines)

    # Construct the full prompt string that guides the LLM
    prompt = """Given Chinese characters with optional OCR, extract word spans.
Return JSON with:
- span: list of indices
- candidates: word + confidence

[
  {
    "span": [0, 1, 2],
    "candidates": [
      { "word": "牛黃散", "confidence": 0.92 },
      { "word": "牛用散", "confidence": 0.35 }
    ]
  }
]

Characters:
""" + joined_sequence

    return prompt

# Run the prompt through the Mistral model via Jsonformer and return the structured result.
def query_model(prompt: str):
    return generate_json(prompt)

