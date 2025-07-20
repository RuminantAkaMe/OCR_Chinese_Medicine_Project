from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def render_raw_json_to_pdf(json_path, pdf_name="word_recognition_output.pdf") -> str:
    """
    Renders the raw content of a JSON file to a PDF file (line by line).
    
    Args:
        json_path (str or Path): Path to the output JSON file.
        pdf_name (str): Filename for the generated PDF.

    Returns:
        str: Full path to the saved PDF.
    """
    json_path = Path(json_path)
    data_dir = json_path.parent
    pdf_path = data_dir / pdf_name

    # Read all lines as text
    with json_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    x = 40
    y = height - 40
    line_height = 14

    for line in lines:
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(x, y, line.strip())
        y -= line_height

    c.save()
    return str(pdf_path)
