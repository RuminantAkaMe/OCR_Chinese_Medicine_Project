"""
🌐 main.py – Einstiegspunkt der FastAPI-Anwendung

Beinhaltet:
- Setup der FastAPI-Instanz mit CORS
- Upload-Endpunkt (/api/upload)
- Operation-Endpunkt (/api/operate)
- Reset-Endpunkt (/api/reset)
- Download-Endpunkt (/api/download/{filename})

Globale Variablen:
- original_file_path → Pfad zur Originaldatei
- processed_file_path → aktueller bearbeiteter Stand

Verwendet:
- `processing.py` für Bildbearbeitung
- `uploaded_files/` als temporäres Datei-Storage
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.processing import process_image
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Globale Pfade
original_file_path: str | None = None
processed_file_path: str | None = None


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global original_file_path, processed_file_path

    # Upload-Verzeichnis leeren
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARN] Datei nicht gelöscht: {e}")

    original_file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(original_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Vorschau direkt möglich machen
    processed_file_path = original_file_path

    return {"filename": file.filename}




@app.post("/api/operate")
async def apply_operation(mode: str = Form(...)):
    global original_file_path, processed_file_path

    if original_file_path is None or not os.path.exists(original_file_path):
        return {"error": "Keine Datei hochgeladen."}

    # Wähle Input: entweder das zuletzt bearbeitete oder das Original
    input_path = processed_file_path or original_file_path

    # Verarbeite Bild
    processed_file_path = process_image(input_path, UPLOAD_DIR, mode)

    return {
        "filename": os.path.basename(processed_file_path)
    }


@app.post("/api/reset")
def reset_processing():
    global processed_file_path

    if original_file_path is not None and os.path.exists(original_file_path):
        processed_file_path = original_file_path
        return {"filename": os.path.basename(original_file_path)}

    return {"error": "Kein Originalbild vorhanden."}




@app.get("/api/download/{filename}")
def download_file(filename: str):
    global processed_file_path

    if processed_file_path is None or not os.path.exists(processed_file_path):
        return {"error": "Kein verarbeiteter Download verfügbar."}

    return FileResponse(
        path=processed_file_path,
        filename=os.path.basename(processed_file_path),
        media_type="application/octet-stream"
    )
