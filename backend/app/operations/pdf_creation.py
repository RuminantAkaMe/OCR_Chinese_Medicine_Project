from PIL import Image
import os

def run(image: Image.Image, output_dir: str) -> str:
    # Dummy: just save the image as a one-page PDF
    filename = "output.pdf"
    output_path = os.path.join(output_dir, filename)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(output_path, "PDF", resolution=100.0)
    return output_path
