# train.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
from transformers import BitsAndBytesConfig
import json

bnb_config = BitsAndBytesConfig(
    #load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

MODEL_PATH = "E:/Software-Projekte/Mistral/Mistral-7B-v0.1"

def load_data(path: Path) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def tokenize(example, tokenizer):
    input_ids = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)["input_ids"]
    label_ids = tokenizer(example["output"], truncation=True, padding="max_length", max_length=512)["input_ids"]
    return {"input_ids": input_ids, "labels": label_ids}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config
    )


    # LoRA Konfiguration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # Daten laden und vorbereiten
    dataset_path = Path(__file__).parent.parent / "word_recognition_src" / "data" / "train.jsonl"
    dataset = load_data(dataset_path)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=["input", "output"])

    # Trainingsparameter
    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()

if __name__ == "__main__":
    main()