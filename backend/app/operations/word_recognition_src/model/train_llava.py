# to train call in bash: $ python backend/app/operations/word_recognition_src/model/train_llava.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# image size must be divisable by 14

# train_llava.py

from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from pathlib import Path
from PIL import Image
import torch
import json

# === Model and processor configuration ===
# Define the path to the pre-trained model you will be fine-tuning
# Path for HPC:
# MODEL_PATH = str(Path(__file__).resolve().parent.parent / "model" / "llava-onevision-qwen2-0.5b-ov-hf")
MODEL_PATH = "E:/Software-Projekte/Llava/llava-onevision-qwen2-0.5b-ov-hf"

# Configure quantization for the model using Bits and Bytes (bnb) for efficient memory use
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Load model in 4-bit precision
    bnb_4bit_use_double_quant=True, # Use double quantization for better precision
    bnb_4bit_quant_type="nf4", # Specify quantization type (normalized float 4-bit)
    bnb_4bit_compute_dtype=torch.float32  # Use 32-bit floating point for computation
)

# Configure LoRA (Low-Rank Adaptation) for efficient model fine-tuning
peft_config = LoraConfig(
    r=8, # Rank of the adaptation matrices
    lora_alpha=32, # Scaling factor for LoRA updates
    lora_dropout=0.1, # Dropout rate during fine-tuning to prevent overfitting
    bias="none", # No bias in LoRA adaptation
    task_type=TaskType.CAUSAL_LM, # The task type (causal language modeling)
    target_modules=["q_proj", "v_proj"] # LoRA will be applied to these layers
)

# === Load dataset from JSONL ===
def load_data(path: Path) -> Dataset:
    """
    Load data from a JSON Lines (JSONL) file and convert it into a Hugging Face Dataset.
    
    Args:
        path (Path): The path to the JSONL file.
        
    Returns:
        Dataset: A Hugging Face Dataset object containing the loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        # Read each line in the file and parse the JSON data
        data = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(data)

# === Preprocess sequence input with multiple images and OCR entries ===
def preprocess(example, processor):
    """
    Preprocess an example for input to the model. This involves converting the images to tensors,
    creating an OCR sequence, and formatting the input and output as required for training.
    
    Args:
        example (dict): A single training example containing input and output data.
        processor (AutoProcessor): A Hugging Face processor to handle tokenization and image processing.
        
    Returns:
        dict: A dictionary containing preprocessed tensors ready for model input.
    """
    images = []
    ocr_sequence = []
    #print(">>> running train_llava from", __file__)
    
    # Process each token in the input to load the corresponding image and OCR data
    for i, token in enumerate(example["input"]):
        img_path = Path(__file__).parent.parent / token["img"]
        image = Image.open(img_path).convert("RGB")
        images.append(image)

        # Build the OCR sequence string
        char = token["ocr"]
        conf = token.get("ocr_confidence", None)
        if conf is not None:
            ocr_sequence.append(f"{i}: {char}({conf:.2f})")
        else:
            ocr_sequence.append(f"{i}: {char}")

    # Build a readable OCR prompt (just for the user text part).
    ocr_prompt = " ".join(ocr_sequence)

    # Ground-truth JSON we want the model to generate as the assistant reply.
    output_json = json.dumps(example["output"], ensure_ascii=False)

    # 1) USER-ONLY messages (for measuring the prompt length).
    messages_prompt = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": img} for img in images] +  # all input images
            [{"type": "text", "text": f"请根据这些图像（{ocr_prompt}）组成一个词语，并给出 JSON 格式的候选词及其范围。"}]
        )
    }]

    # 2) FULL conversation = USER + ASSISTANT(target JSON).
    messages_full = messages_prompt + [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": output_json}
        ]
    }]

    # Manually process images to get pixel tensors in the exact format the vision tower expects.
    image_tensors = []
    image_sizes = []
    for img in images:
        processed = processor.image_processor(img, return_tensors="pt")
        tensor = processed["pixel_values"].squeeze(0)   # (C, H, W)
        image_tensors.append(tensor)
        image_sizes.append(list(tensor.shape[-2:]))     # record (H, W) after processing

    # Encode the USER-ONLY prompt with a generation tag to get its token length.
    enc_prompt = processor.apply_chat_template(
        messages_prompt,
        add_generation_prompt=True,   # add assistant-begin tag at the end of the prompt
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Encode the FULL conversation; these tokens are fed to the model during training.
    enc_full = processor.apply_chat_template(
        messages_full,
        add_generation_prompt=False,  # assistant answer is already included
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Inputs the model will see.
    input_ids = enc_full["input_ids"].squeeze(0)
    attention_mask = enc_full["attention_mask"].squeeze(0)

    # Labels must match input_ids shape. We mask the entire prompt (user side) to -100
    # so loss is computed only on the assistant part (the target JSON).
    labels = input_ids.clone()
    prompt_len = enc_prompt["input_ids"].shape[-1]  # token length up to (and including) assistant-begin
    labels[:prompt_len] = -100 # no loss computation for user input

    # Stack all image tensors of this sample; collator will handle batching.
    pixel_values = torch.stack(image_tensors)

    # 1) Wie viele Bilder im Prompt?
    n_tokens = sum(1 for c in messages_full[0]["content"] if c["type"] == "image")

    # 2) Wie viele Bilder gehen als Features rein?
    n_feats = pixel_values.shape[0]  # nach torch.stack: (num_images, C, H, W)

    assert n_tokens == len(images) == n_feats, \
        f"Mismatch: tokens={n_tokens}, images={len(images)}, features_images={n_feats}"

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
    }

# === Custom collator ===
def data_collator(features):
    """
    Custom collator for batching the data during training. Pads the sequences and images to ensure 
    that all input sequences and image tensors in the batch have the same shape.
    
    Args:
        features (list): A list of preprocessed examples to batch together.
        
    Returns:
        dict: A dictionary containing batched and padded input tensors ready for model input.
    """
    input_ids = [torch.as_tensor(f["input_ids"]) for f in features]
    labels = [torch.as_tensor(f["labels"]) for f in features]
    attention_mask = [torch.as_tensor(f["attention_mask"]) for f in features]

    pixel_values_list = []
    image_sizes_list = []
    for f in features:
        pv = f["pixel_values"]
        if isinstance(pv, list):
            pv = torch.stack([torch.as_tensor(p) for p in pv])
        else:
            pv = torch.as_tensor(pv)

        sizes = torch.as_tensor(f["image_sizes"])
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
            sizes = sizes.unsqueeze(0)
        elif pv.dim() > 3:
            pv = pv.view(-1, *pv.shape[-3:])
            sizes = sizes.view(-1, 2)
        else:
            raise ValueError(f"Unexpected pixel_values shape: {pv.shape}")
        pixel_values_list.append(pv)
        image_sizes_list.append(sizes)

    # Pad the sequences to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Concatenate all images across the batch. This avoids introducing dummy
    # image entries with zero sizes, which previously caused division-by-zero
    # errors when computing patch counts.
    pixel_values = torch.cat(pixel_values_list, dim=0)
    image_sizes = torch.cat(image_sizes_list, dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes
    }

# === Training ===
def main():
    """
    The main function for training the model. It loads the dataset, preprocesses the data,
    initializes the model, and sets up the training loop using Hugging Face's Trainer API.
    """
    # Initialize the processor and model from the pre-trained model path
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config) # Apply LoRA to the model

    # Load and preprocess the dataset
    dataset_path = Path(__file__).parent.parent / "data" / "train.jsonl"
    dataset = load_data(dataset_path)
    dataset = dataset.map(lambda x: preprocess(x, processor), remove_columns=dataset.column_names)

    # Set up training arguments
    args = TrainingArguments(
        output_dir="./checkpoints", # Output directory for model checkpoints
        per_device_train_batch_size=1, # Batch size per device
        gradient_accumulation_steps=4, # Accumulate gradients over 4 steps
        num_train_epochs=3, # Train for 3 epochs
        learning_rate=2e-4, # Learning rate
        logging_dir="./logs", # Directory to store logs
        logging_steps=10, # Log training progress every 10 steps
        save_strategy="epoch", # Save the model checkpoint at the end of each epoch
        bf16 = True,  # oder fp16=True wenn keine bf16-Unterstützung
        gradient_checkpointing = True,
        optim = "paged_adamw_8bit",
        remove_unused_columns = False,
    )

    # Initialize the Trainer class with the model, arguments, dataset, and collator
    trainer = Trainer(
        model=model, # The model to train
        args=args, # Training arguments
        train_dataset=dataset, # The preprocessed training dataset
        tokenizer=processor.tokenizer, # The tokenizer to use for processing the input text
        data_collator=data_collator # Custom collator to handle padding of inputs and images
    )

    # Start the training loop
    trainer.train()

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()

