from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
from transformers import BitsAndBytesConfig
import json
import torch

# Modellpfad von Hugging Face
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# BitsAndBytes: Quantisierung aktivieren
bnb_config = BitsAndBytesConfig(
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Daten laden aus .jsonl
def load_data(path: Path) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

# Prompt-Format wie bei Inferenz
def tokenize(example, tokenizer):
    prompt = f"<|user|>\n<|image|>\n{example['input']}\n<|end|>\n<|assistant|>\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def main():
    # Tokenizer und Modell laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
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

    # Dataset vorbereiten
    dataset_path = Path(__file__).parent.parent / "word_recognition_src" / "data" / "train.jsonl"
    dataset = load_data(dataset_path)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=["input", "output"])

    # Trainingsargumente
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
