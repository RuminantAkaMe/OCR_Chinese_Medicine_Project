from PIL import Image, ImageDraw

def run(image: Image.Image) -> Image.Image:
    # Dummy: draw a red rectangle (simulate a detection)
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 150, 150], outline="red", width=5)
    return image
