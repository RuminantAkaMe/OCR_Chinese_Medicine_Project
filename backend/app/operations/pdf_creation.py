# for creation call in bash:
# $ python backend/app/operations/pdf_creation.py --output_dir backend/uploaded_files
# DONT FORGET to activate your python environment:
# source backend/.venv310/Scripts/activate

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

def load_chinese_font(font_size: int) -> ImageFont.ImageFont:
    """
    Attempts to load a font that supports Chinese characters.
    Tries common font locations, returns a fallback if none found.

    Args:
        font_size (int): Desired font size.

    Returns:
        ImageFont.ImageFont: Loaded font that supports Chinese text.
    """
    possible_fonts = [
        Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("arial.ttf"),
    ]
    for font_path in possible_fonts:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except:
                continue
    return ImageFont.load_default()

def get_color_for_confidence(conf: float) -> str:
    """
    Maps a confidence value to a color for bounding boxes.

    Args:
        conf (float): Confidence score from the model.

    Returns:
        str: Color name for drawing the rectangle.
    """
    if conf >= 0.7:
        return "green"
    elif conf >= 0.4:
        return "orange"
    else:
        return "red"

def run(output_dir: str) -> str:
    """
    Generates a visual PDF showing:
    - OCR-segmented character images
    - OCR labels under each image
    - Predicted word spans from the model, with bounding boxes
    - Confidence-based coloring for each prediction

    Args:
        output_dir (str): Directory where the output PDF will be saved.

    Returns:
        str: Path to the generated PDF file.
    """
    # Define project-relative paths
    operations_dir = Path(__file__).resolve().parent
    data_dir = operations_dir / "word_recognition_src" / "data"
    sequence_path = data_dir / "sequence.json"
    word_recognition_path = data_dir / "output" / "output_full.json"

    # Load sequence and prediction data
    with sequence_path.open(encoding="utf-8") as f:
        sequence = json.load(f)

    with word_recognition_path.open(encoding="utf-8") as f:
        predictions = json.load(f)

    # Get dimensions from first character image
    first_img = Image.open(data_dir / sequence[0]["img"])
    char_width, char_height = first_img.size

    # Layout settings
    spacing = 10
    margin = 20
    font_size = 18
    max_chars_per_row = 20
    font = load_chinese_font(font_size)

    # Compute image dimensions
    num_rows = (len(sequence) + max_chars_per_row - 1) // max_chars_per_row
    total_width = max_chars_per_row * (char_width + spacing) + 2 * margin
    total_height = num_rows * (char_height + 100) + margin

    # Create the base PDF image
    pdf_img = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(pdf_img)

    # Draw character images and their OCR labels
    positions = []  # track where each character is placed
    for idx, entry in enumerate(sequence):
        row = idx // max_chars_per_row
        col = idx % max_chars_per_row

        x = margin + col * (char_width + spacing)
        y = margin + row * (char_height + 100)

        char_img = Image.open(data_dir / entry["img"])
        pdf_img.paste(char_img, (x, y))

        draw.text((x, y + char_height + 5), entry["ocr"], fill="black", font=font)
        positions.append((x, y))

    # Draw prediction boxes with confidence labels
    for pred in predictions:
        window_ids = pred.get("window_ids", [])
        if not window_ids:
            continue  # skip if no span info

        start_idx = window_ids[0]
        end_idx = window_ids[-1]
        if start_idx >= len(positions) or end_idx >= len(positions):
            print(f"[WARN] Skipping: window [{start_idx}, {end_idx}] out of range for {len(positions)} characters.")
            continue  # out of bounds

        word = pred["output"]["text"]
        conf = pred["output"]["confidence"]

        # Group character positions by row (y coordinate)
        from collections import defaultdict
        row_boxes = defaultdict(list)

        for idx in window_ids:
            if idx >= len(positions):
                continue
            x, y = positions[idx]
            row_boxes[y].append(x)

        if not row_boxes:
            continue

        # Get model output info
        word = pred["output"]["text"]
        conf = pred["output"]["confidence"]
        color = get_color_for_confidence(conf)

        # For each row, draw a separate box
        for row_y, x_list in row_boxes.items():
            x0 = min(x_list)
            x1 = max(x_list) + char_width
            box_y0 = row_y
            box_y1 = row_y + char_height

            draw.rectangle([x0, box_y0, x1, box_y1], outline=color, width=3)
            # Only draw the label once (above the first box)
            if row_y == min(row_boxes.keys()):
                label = f"{word} ({conf:.2f})"
                draw.text((x0, box_y0 - font_size - 5), label, fill=color, font=font)



        if x1 <= x0:
            print(f"[WARN] Skipping invalid box: x0={x0}, x1={x1}, word='{word}'")
            continue  # skip degenerate box

        color = get_color_for_confidence(conf)
        # Draw filled background with low opacity using RGBA mode
        fill_color = {
            "red": (255, 0, 0, 50),
            "orange": (255, 165, 0, 50),
            "green": (0, 128, 0, 50)
        }.get(color, (0, 0, 0, 50))

        # Draw filled rectangle on overlay
        overlay = Image.new("RGBA", pdf_img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x0, box_y0, x1, box_y1], fill=fill_color)

        # Combine overlay with base image
        pdf_img = Image.alpha_composite(pdf_img.convert("RGBA"), overlay).convert("RGB")

        # Then draw border
        draw = ImageDraw.Draw(pdf_img)
        draw.rectangle([x0, box_y0, x1, box_y1], outline=color, width=3)


        label = f"{word} ({conf:.2f})"
        draw.text((x0, box_y0 - font_size - 5), label, fill=color, font=font)

    # Save final image as PDF
    output_path = Path(output_dir) / "output.pdf"
    pdf_img.save(output_path, "PDF", resolution=100.0)
    return str(output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate annotated PDF from OCR and word recognition output.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output PDF will be saved."
    )
    args = parser.parse_args()

    pdf_path = run(args.output_dir)
    print(f"PDF saved to: {pdf_path}")
