# to train call in bash: $ python backend/app/operations/word_recognition_src/model/train.py
# DONT FORGET to activate your python environment
# image size must be divisable by 14

from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from pathlib import Path
from PIL import Image
import torch
import json
import numpy as np

# === Model and processor configuration ===
MODEL_PATH = "E:/Software-Projekte/SmolVLM/SmolVLM2-2.2B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

# === Load dataset from JSONL ===
def load_data(path: Path) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(data)

# === Preprocess a single training example ===
def preprocess(example, processor):
    image_path = Path(__file__).parent.parent / example["img"]
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)  # [3, H, W]

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"OCR: {example['ocr']}"},
            {"type": "image", "image": image},
            {"type": "text", "text": "What word do these chinese characters represent?"}
        ]
    }]

    encoding = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors=None
    )

    def to_tensor(val):
        if isinstance(val, list):
            return torch.tensor(val, dtype=torch.long if all(isinstance(x, int) for x in val) else torch.float)
        elif isinstance(val, np.ndarray):
            return torch.tensor(val)
        elif isinstance(val, torch.Tensor):
            return val
        elif isinstance(val, (float, np.floating)):
            return torch.tensor([val], dtype=torch.float)
        elif isinstance(val, (int, np.integer)):
            return torch.tensor([val], dtype=torch.long)
        else:
            raise TypeError(f"Cannot convert type {type(val)} to tensor")

    encoding = {k: to_tensor(v) for k, v in encoding.items()}

    input_ids = encoding["input_ids"].squeeze()
    output_ids = processor.tokenizer(
        example["output"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )["input_ids"].squeeze(0)

    labels = torch.cat([input_ids, output_ids], dim=0)[:512]
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values  # [3, H, W]
    }

# === Custom collator ===
def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

    pixel_values = []
    for f in features:
        pv = f["pixel_values"]
        if isinstance(pv, list):
            pv = torch.tensor(pv)
        if pv.dim() == 3:  # [3, H, W]
            pv = pv.unsqueeze(0)  # → [1, 3, H, W]
        elif pv.dim() != 4:
            raise ValueError(f"Unexpected pixel_value shape: {pv.shape}")
        pixel_values.append(pv)

    # Final shape: [B, 1, 3, H, W]
    pixel_values = torch.stack(pixel_values, dim=0).to(torch.float32)

    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "pixel_values": pixel_values
    }

# === Training ===
def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float32,
        quantization_config=bnb_config
    )
    model = get_peft_model(model, peft_config)

    dataset_path = Path(__file__).parent.parent / "data" / "train.jsonl"
    dataset = load_data(dataset_path)
    dataset = dataset.map(lambda x: preprocess(x, processor), remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()



