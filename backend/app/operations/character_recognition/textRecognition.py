import os
from paddleocr import TextRecognition

def ensure_directory(path: str) -> None:
    """Ensure the specified directory exists."""
    os.makedirs(path, exist_ok=True)

def initialize_model(model_name: str) -> TextRecognition:
    """Initialize and return the OCR model for the given model name."""
    return TextRecognition(model_name=model_name)

def predict_text(model: TextRecognition, image_path: str, batch_size: int = 1):
    """Run text recognition (prediction) on the image."""
    return model.predict(input=image_path, batch_size=batch_size)

def save_results(output, image_path: str, save_img_path: str) -> None:
    """Process and save the recognition results."""
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    for res in output:
        res.print()  # Debug: print result to console
        json_path = os.path.join(save_img_path, f"{img_name}.json")
        res.save_to_json(save_path=json_path)
        print(f"Saved JSON to {json_path}")

def recognize_and_save(image_path: str, save_img_path: str, model_name: str) -> None:
    """
    Recognizes text in the specified image using the given model and saves the results.
    
    Args:
        image_path (str): Path to the input image.
        save_img_path (str): Directory where output files will be saved.
        model_name (str): Identifier of the model to use for recognition.
    """
    ensure_directory(save_img_path)
    model = initialize_model(model_name)
    output = predict_text(model, image_path)
    save_results(output, image_path, save_img_path)


