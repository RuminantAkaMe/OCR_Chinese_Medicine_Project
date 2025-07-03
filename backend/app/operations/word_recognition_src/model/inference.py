# for inference call in bash: $ python backend/app/operations/word_recognition_src/model/inference.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# inference.py 

import torch
from torch.nn.functional import log_softmax
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from pathlib import Path
import json

def generate_candidates_with_confidence(model, processor, input_sequence, max_new_tokens=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generates word predictions with associated confidence scores based on an OCR-character image sequence.

    Args:
        model: The SmolVLM model for multimodal image-text-to-text inference.
        processor: The associated processor to tokenize and format the input.
        input_sequence: A list of dicts, each with 'ocr', 'ocr_confidence', and 'img' keys.
        max_new_tokens: Maximum number of tokens to generate.
        device: "cuda" or "cpu" depending on hardware availability.

    Returns:
        A dictionary containing:
            - text: The predicted word
            - confidence: A single float confidence score
            - tokens: Generated tokens
            - logprobs: Log-probabilities of generated tokens
    """
    base_dir = Path(__file__).parent.parent  # Points to directory containing `data` folder
    images = []
    ocr_sequence = []

    # Load each character image and record its OCR text
    for token in input_sequence:
        img_path = base_dir / "data" / token["img"]
        image = Image.open(img_path).convert("RGB")
        images.append(image)

        char = token["ocr"]
        # Optional: include OCR confidence in prompt
        conf = token.get("ocr_confidence", None)
        ocr_sequence.append(f"{char}({conf:.2f})" if conf is not None else char)

    # Construct multimodal prompt: alternating OCR text and images
    content = []
    for token, img in zip(input_sequence, images):
        content.append({"type": "text", "text": f"OCR: {token['ocr']}"})
        content.append({"type": "image", "image": img})

    # Final instruction for the model
    content.append({"type": "text", "text": "What word do these chinese characters represent?"})

    messages = [{"role": "user", "content": content}]

    # Tokenize inputs using chat template
    chat_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Ensure pixel_values are float16 if using a float16 model on CUDA
    if "pixel_values" in chat_inputs and device == "cuda":
        chat_inputs["pixel_values"] = chat_inputs["pixel_values"].half()

    chat_inputs = chat_inputs.to(device)

    # Generate output using model
    with torch.no_grad():
        output = model.generate(
            **chat_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Extract only the newly generated part (remove input prompt tokens)
    generated_ids = output.sequences[0][chat_inputs["input_ids"].shape[-1]:]
    # Convert to human-readable tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids)
    scores = output.scores

    # Compute token-level log-probabilities
    logprobs = [
        log_softmax(score_dist.squeeze(0), dim=-1)[token_id].item()
        for score_dist, token_id in zip(scores, generated_ids)
    ]
    # Mean of log-probs → aggregate confidence
    mean_logprob = sum(logprobs) / len(logprobs)
    confidence = torch.exp(torch.tensor(mean_logprob)).item()

    # Decode generated token IDs to final string output
    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": output_text,
        "confidence": confidence,
        "tokens": tokens,
        "logprobs": logprobs
    }

if __name__ == "__main__":
    # Local path to the SmolVLM model folder
    MODEL_PATH = "E:/Software-Projekte/SmolVLM/SmolVLM2-2.2B-Instruct"
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        _attn_implementation="eager"
    ).to("cuda")

    # Example OCR character sequence
    example_input = [
        { "id": 0, "ocr": "牛", "ocr_confidence": 0.69, "img": "input/000.png" },
        { "id": 1, "ocr": "黃", "ocr_confidence": 0.41, "img": "input/001.png" },
        { "id": 2, "ocr": "散", "ocr_confidence": 0.77, "img": "input/002.png" }
    ]

    # Run inference
    result = generate_candidates_with_confidence(model, processor, example_input)
    # Pretty-print output
    print(json.dumps(result, ensure_ascii=False, indent=2))
'''
What is a Log Probability in Language Models:

In auto-regressive language models (like SmolVLM), during generation, each token is predicted based on previous context.
The model outputs a logit vector (unnormalized scores) over the entire vocabulary at each generation step.
These logits are transformed into probabilities using a softmax function.


The log-probability of a token is simply: log(P(t_i | t_1, t_2,..., t_i-1))

These log-probs are useful because:
They are additive: the log probability of a sequence is the sum of individual token log-probs.
They avoid numerical instability of multiplying small probabilities.
They're efficient to compute with log_softmax.

The confidence score is not directly predicted by the model. Its an aggregate heuristic computed as:
Calculate log-probability of each generated token at its respective decoding step.
Compute the mean log-probability across the sequence.
Convert this mean log-prob to a probability using the exponential function.
--> This is the geometric mean of the token probabilities — a common method to compute sentence-level confidence.

This confidence measure reflects how likely the model thinks its own output is — it does not measure correctness.
It's affected by:
Repetition (models may assign high prob to generic tokens)
Length (shorter outputs often get higher confidence)
Vocabulary granularity (some Chinese words may span multiple tokens)
It's not comparable across models unless they use the same tokenizer and temperature.
'''

'''
{
  "text": " 水",
  "confidence": 0.2998807430267334,
  "tokens": [
    "Ġæ",
    "°",
    "´",
    "<end_of_utterance>"
  ],
  "logprobs": [
    -1.8279743194580078,
    -2.4783926010131836,
    -0.4239494800567627,
    -0.08716503530740738
  ]
}

'''