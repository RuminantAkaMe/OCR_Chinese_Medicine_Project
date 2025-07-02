from paddleocr import TextLineOrientationClassification
from PIL import Image
import json
import os

model = TextLineOrientationClassification(
    model_name="PP-LCNet_x0_25_textline_ori",
    model_dir="./PP-LCNet_x0_25_textline_ori_infer/"
)

def rotate_image_by_orientation(image_path: str, output_path: str = None) -> str:
    """
    Predicts the orientation of the text line in the image and rotates it back if needed.
    Args:
        image_path (str): Path to the image file.
        output_path (str, optional): Where to save the rotated image. If None, overwrites original.
    Returns:
        str: The path to the rotated (or original) image.
    """
    output = model.predict(image_path, batch_size=1)
    if not output:
        raise ValueError("No prediction result returned.")

    label = output[0].label_names[0] if output[0].label_names else None

    img = Image.open(image_path)
    rotated = False

    if label == "180_degree":
        img = img.rotate(180, expand=True)
        rotated = True
    elif label == "90_degree":
        img = img.rotate(-90, expand=True)
        rotated = True
    elif label == "270_degree":
        img = img.rotate(-270, expand=True)
        rotated = True

    save_path = output_path if output_path else image_path
    img.save(save_path)
    return save_path

# Example usage:
# rotated_path = rotate_image_by_orientation("./model/textline_rot180_demo.jpg", "./output/rotated_demo.jpg")