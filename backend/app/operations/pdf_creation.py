from PIL import Image
import os

def run(output_dir: str) -> str:
    # Dummy: just save the image as a one-page PDF
    filename = "output.pdf"
    output_path = os.path.join(output_dir, filename)

    # Create a blank white image (e.g., A4 at 72 DPI ≈ 595x842 px)
    image = Image.new("RGB", (595, 842), color="white")

    image.save(output_path, "PDF", resolution=100.0)
    return output_path
