from PIL import Image

def run(image: Image.Image) -> Image.Image:
    """
    Dummy preprocessing function.
    For now: just convert to grayscale.
    """
    return image.convert("L")
