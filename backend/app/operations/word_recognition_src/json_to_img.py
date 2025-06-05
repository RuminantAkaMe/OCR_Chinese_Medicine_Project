from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def render_raw_json_to_image(json_path, image_name="word_recognition_output.png") -> str:
    """
    Renders the raw content of a JSON file to a PNG image (line by line).
    
    Args:
        json_path (str or Path): Path to the output JSON file.
        image_name (str): Filename for the generated image.

    Returns:
        str: Full path to the saved image.
    """
    json_path = Path(json_path)
    data_dir = json_path.parent
    image_path = data_dir / image_name

    # Read entire content as plain text
    with json_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Image size estimation
    width = 1600
    line_height = 20
    height = 40 + len(lines) * line_height

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    y = 10
    for line in lines:
        draw.text((10, y), line.rstrip(), fill="black", font=font)
        y += line_height

    img.save(image_path)
    return str(image_path)


