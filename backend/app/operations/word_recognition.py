from PIL import Image, ImageOps

def run(image: Image.Image) -> Image.Image:
    # Dummy: invert the image colors to simulate "recognition"
    return ImageOps.invert(image)
