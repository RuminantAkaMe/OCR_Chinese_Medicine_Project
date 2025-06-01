from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from jsonformer import Jsonformer
from pathlib import Path
import json
import torch

# Set default data type to float16 to reduce memory usage and computation cost.
# Note: This can also reduce numerical precision.
torch.set_default_dtype(torch.float16)

# Path to the locally saved Mistral model.
MODEL_PATH = "E:/Software-Projekte/Mistral/Mistral-7B-v0.1"

# Configuration for quantization using BitsAndBytes.
bnb_config = BitsAndBytesConfig(
    # 4-bit loading is commented out; only double quantization and NF4 are active.
    # This helps compress model weights and reduce GPU memory use.
    #load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# Load the tokenizer associated with the model.
# Use `use_fast=False` to load the slow (more compatible) tokenizer variant.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# Load the causal language model with quantization and automatic device placement.
# `offload_folder` allows parts of the model to be temporarily offloaded to CPU storage.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    #offload_buffers=True
    offload_folder="offload",
)

# Load the expected output structure for the JSONformer using a JSON schema file.
def load_schema():
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Generate structured JSON output using the model and JSONformer.
# Input: a natural language prompt describing the task.
# Output: structured list from the model matching the schema.
def generate_json(prompt: str) -> list:

    # Clear unused CUDA memory before inference (helps reduce fragmentation).
    torch.cuda.empty_cache()

    schema = load_schema()

    # Initialize JSONformer with model, tokenizer, prompt and schema.
    generator = Jsonformer(
        model=model,
        tokenizer=tokenizer,
        json_schema=schema,
        prompt=prompt,
        # Optional: limit output length if needed.
        #max_new_tokens=64,
    )

    # Call the generator and return the "results" field from its output.
    output = generator()
    return output["results"]



