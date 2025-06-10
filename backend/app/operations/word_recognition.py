from PIL import Image
import os

def run(_: Image.Image) -> Image.Image:
    base_path = os.path.dirname(__file__)
    image_path = os.path.join(base_path, "word_recognition_src", "output", "output.png")
    return Image.open(image_path)

