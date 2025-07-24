# for creation call in bash:
# $ python backend/app/operations/pdf_creation.py --output_dir backend/uploaded_files
# DONT FORGET to activate your python environment:
# source backend/.venv310/Scripts/activate

"""
Script for generating an annotated PDF from a sequence of OCR-processed character images
and corresponding word recognition predictions.

Each page shows character images in a row, overlays prediction bounding boxes with
confidence-based coloring, and includes a legend summarizing recognized words and scores.

Usage (from project root):
$ source backend/.venv310/Scripts/activate
$ python backend/app/operations/pdf_creation.py --output_dir backend/uploaded_files
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
#import hashlib
#from collections import defaultdict


def load_chinese_font(font_size: int) -> ImageFont.ImageFont:
    """
    Attempts to load a Chinese-capable font to render characters in the PDF.
    Falls back to system fonts and finally PIL's default if none found.

    Args:
        font_size (int): Desired font size.

    Returns:
        ImageFont.ImageFont: A loaded font object.
    """
    possible_fonts = [
        Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.ttf",  # Project font
        Path("C:/Windows/Fonts/msyh.ttc"),  # Microsoft YaHei
        Path("C:/Windows/Fonts/simsun.ttc"),  # SimSun
        Path("arial.ttf"),  # Arial fallback
    ]
    for font_path in possible_fonts:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except:
                continue
    return ImageFont.load_default()


def get_confidence_color(conf: float) -> str:
    """
    Maps a prediction confidence score to a display color.

    Args:
        conf (float): Confidence value between 0.0 and 1.0

    Returns:
        str: A color string ("green", "orange", "red")
    """
    if conf >= 0.7:
        return "green"  # High confidence
    elif conf >= 0.4:
        return "orange"  # Medium confidence
    else:
        return "red"  # Low confidence


def run(output_dir: str) -> str:
    """
    Main function that loads OCR character images and word recognition predictions,
    arranges them into annotated PDF pages, and saves the final document.

    Args:
        output_dir (str): Directory path where the output PDF will be stored.

    Returns:
        str: Full path to the generated PDF file.
    """
    # Locate necessary paths
    operations_dir = Path(__file__).resolve().parent
    data_dir = operations_dir / "word_recognition_src" / "data"
    sequence_path = data_dir / "sequence.json"
    word_recognition_path = data_dir / "output" / "output_full.json"

    # Load character image sequence and prediction data
    with sequence_path.open(encoding="utf-8") as f:
        sequence = json.load(f)
    with word_recognition_path.open(encoding="utf-8") as f:
        predictions = json.load(f)

    # Determine image size by loading the first character image
    first_img = Image.open(data_dir / sequence[0]["img"])
    char_width, char_height = first_img.size

    # Layout configuration
    spacing = 10  # Horizontal space between character images
    margin = 20  # Margin around content
    font_size = 18
    font = load_chinese_font(font_size)
    legend_line_height = font_size + 6  # Spacing between legend lines

    max_chars_per_row = len(sequence)  # Force all characters into one row
    chars_per_page = len(sequence)  # Entire sequence fits on one "virtual" page

    pages = []  # Holds each image page that will be saved into PDF
    total_chars = len(sequence)
    total_pages = (total_chars + chars_per_page - 1) // chars_per_page  # Usually 1

    # Iterate through each "page" of characters (usually one)
    for page_index in range(total_pages):
        char_start_idx = page_index * chars_per_page
        char_end_idx = min(char_start_idx + chars_per_page, total_chars)
        page_sequence = sequence[char_start_idx:char_end_idx]

        # Extract predictions that appear in this page's character range
        page_predictions = [
            p for p in predictions if char_start_idx <= p.get("window_ids", [0])[0] < char_end_idx
        ]
        legend_height = len(page_predictions) * legend_line_height + 2 * margin

        # Compute page dimensions
        page_width = max_chars_per_row * (char_width + spacing) + 2 * margin
        page_height = (char_height + 40) + legend_height + 2 * margin

        # Create blank image canvas for the page
        pdf_img = Image.new("RGBA", (page_width, page_height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(pdf_img)
        positions = []  # To store (x, y) of each character

        # Draw character images and corresponding OCR text
        for i, entry in enumerate(page_sequence):
            col = i % max_chars_per_row
            x = margin + col * (char_width + spacing)
            y = margin
            char_img = Image.open(data_dir / entry["img"])
            pdf_img.paste(char_img, (x, y))
            draw.text((x, y + char_height + 2), entry["ocr"], fill="black", font=font)
            positions.append((x, y))

        legend_entries = []  # To collect data for the legend
        alt = True  # Flag for alternating top/bottom label placement

        # Draw bounding boxes and collect legend information
        for i, pred in enumerate(page_predictions):
            window_ids = pred.get("window_ids", [])
            if not window_ids:
                continue

            # Get indices relative to current page
            local_indices = [idx - char_start_idx for idx in window_ids if char_start_idx <= idx < char_end_idx]
            if not local_indices or any(idx >= len(positions) for idx in local_indices):
                continue

            # Skip predictions that span multiple rows (not expected in single-row layout)
            row_numbers = set(idx // max_chars_per_row for idx in local_indices)
            if len(row_numbers) > 1:
                continue

            # Calculate bounding box for prediction
            xs = [positions[idx][0] for idx in local_indices]
            ys = [positions[idx][1] for idx in local_indices]
            x0 = min(xs)
            x1 = max(xs) + char_width
            y0 = ys[0]

            conf = pred["output"].get("confidence", 0.0)
            color = get_confidence_color(conf)

            # Calculate box vertical position depending on alternation
            offset = 5
            y_top = y0 - offset if alt else y0
            y_bottom = y0 + char_height + (offset if not alt else 0)

            draw.rectangle([x0, y_top, x1, y_bottom], outline=color, width=2)  # Draw the box
            draw.text((x0 + 3, y_top + 2 if alt else y_bottom - font_size - 2), f"{i+1:02d}", fill=color, font=font)

            alt = not alt  # Flip alternation for next box

            # Add legend entry
            word = pred["output"].get("text", "")
            legend_entries.append((i + 1, word, window_ids[0], conf, color))

        # Draw legend below character row
        legend_y = page_height - legend_height + 10
        for j, (num, word, start, conf, color) in enumerate(legend_entries):
            label = f"{num:02d}. {word} (start={start}, conf={conf:.2f})"
            draw.text((margin, legend_y + j * legend_line_height), label, fill=color, font=font)

        # Layout slicing logic — splits wide rows into segments if too long
        image_row_height = char_height + 40 + margin
        full_char_width = char_width + spacing
        chars_per_slice = 20  # Number of characters per horizontal slice
        slice_width = chars_per_slice * full_char_width
        page_height = image_row_height + legend_height + margin
        num_slices = (len(page_sequence) + chars_per_slice - 1) // chars_per_slice

        for i in range(num_slices):
            # Calculate crop box
            x0 = margin + i * slice_width
            x1 = x0 + slice_width - spacing

            # Crop upper image row
            img_crop = pdf_img.crop((x0, 0, x1, image_row_height))

            # Crop full legend
            legend_crop = pdf_img.crop((0, image_row_height, pdf_img.width, pdf_img.height))

            # Create final composed page
            page = Image.new("RGB", (slice_width, page_height), "white")
            page.paste(img_crop, (0, 0))
            page.paste(legend_crop, (0, image_row_height))
            pages.append(page)

    # Save all image pages into a single PDF
    output_path = Path(output_dir) / "output.pdf"
    pages[0].save(output_path, save_all=True, append_images=pages[1:], format="PDF", resolution=100.0)

    return str(output_path)


if __name__ == "__main__":
    # Handle CLI execution
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate annotated PDF from OCR and word recognition output."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output PDF will be saved."
    )
    args = parser.parse_args()
    pdf_path = run(args.output_dir)
    print(f"PDF saved to: {pdf_path}")
