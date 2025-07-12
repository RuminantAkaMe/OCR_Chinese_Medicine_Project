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

    # Process each token in the input to load the corresponding image and OCR data
    for token in example["input"]:
        img_path = Path(__file__).parent.parent / token["img"]
        image = Image.open(img_path).convert("RGB")
        images.append(image)

        # Build the OCR sequence string
        char = token["ocr"]
        conf = token.get("ocr_confidence", None)
        ocr_sequence.append(f"{char}({conf:.2f})" if conf is not None else char)

    ocr_prompt = " ".join(ocr_sequence)

    # Prepare the message structure for the model
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": img} for img in images] +
            [{"type": "text", "text": f"请根据这些图像（{ocr_prompt}）组成一个词语，并给出 JSON 格式的候选词及其范围。"}]
        )
    }]

    # Apply the processor's chat template to format the messages for the model
    # Returning tensors here ensures pixel values and image sizes are computed
    # exactly as expected by the model's vision tower.
    encoding = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].view(-1)


    # Prepare the output labels by encoding them
    output_json = json.dumps(example["output"], ensure_ascii=False)
    output_ids = processor.tokenizer(
        output_json,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )["input_ids"].squeeze(0)
    if output_ids.dim() > 1:
        output_ids = output_ids.view(-1)

    # Concatenate the input and output IDs, truncate to a maximum length of 512
    labels = torch.cat([input_ids, output_ids], dim=0)[:512]
    attention_mask = torch.ones_like(input_ids)

    # Manually process the images so we can record the exact tensor shapes
    # used during training. This ensures the patch counts match the model's
    # vision tower expectations.
    image_tensors = []
    image_sizes = []
    for img in images:
        processed = processor.image_processor(img, return_tensors="pt")
        tensor = processed["pixel_values"].squeeze(0)
        image_tensors.append(tensor)
        # record the tensor height/width after processing
        image_sizes.append(list(tensor.shape[-2:]))

    pixel_values = torch.stack(image_tensors)

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
        torch_dtype=torch.float32,
        quantization_config=bnb_config
    )
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
        fp16=False # Whether to use 16-bit precision (disabled here for stability)
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

