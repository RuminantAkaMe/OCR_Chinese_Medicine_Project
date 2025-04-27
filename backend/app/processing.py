"""
🧠 processing.py – Bildverarbeitung mit Pillow

Beinhaltet:
- `process_image(input_path, output_dir, mode)`:
    Führt eine einfache Bildoperation aus, abhängig vom Modus.
    Unterstützte Modi:
    - resize → Bild wird auf 300x300 verkleinert
    - rotate → Bild wird um 90° gedreht
    - grayscale → Bild wird in Graustufen konvertiert
    - fallback → resize bei unbekanntem Modus

Verwendet:
- Pillow (`from PIL import Image`)
- Gibt Pfad zur verarbeiteten Datei zurück
"""

from PIL import Image
import os

def process_image(input_path: str, output_dir: str, mode: str = "resize") -> str:
    try:
        image = Image.open(input_path)
        filename = os.path.basename(input_path)
        output_filename = f"{mode}_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        if mode == "resize":
            image.thumbnail((300, 300))
        elif mode == "rotate":
            image = image.rotate(90)
        elif mode == "grayscale":
            image = image.convert("L")
        else:
            print(f"[WARN] Unbekannter Modus '{mode}', Standard: resize")
            image.thumbnail((300, 300))

        image.save(output_path)
        return output_path

    except Exception as e:
        print(f"[ERROR] Bildverarbeitung fehlgeschlagen: {e}")
        return input_path  # Notlösung: Original zurückgeben
