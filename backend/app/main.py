"""
🌐 main.py – Entry point of the FastAPI application

Includes:
- Setup of the FastAPI instance with CORS
- Upload endpoint (/api/upload)
- Processing endpoints (preprocessing, detection, segmentation, recognition, PDF creation)
- Reset endpoint (/api/reset)
- Download endpoint (/api/download/{filename})
- Preview endpoint (/api/preview/{filename})

Global variables:
- original_file_path → path to the original uploaded file
- processed_file_path → path to the currently processed file

Uses:
- `operations/` folder for separate image processing steps
- `uploaded_files/` folder for temporary file storage
"""


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.operations import character_detection 
from app.operations import character_segmentation 
from app.operations import character_recognition
from app.operations import word_recognition
from app.operations import pdf_creation
import os
import shutil
from fastapi.responses import FileResponse
import mimetypes
from fastapi import HTTPException

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

# Global paths
original_file_path: str | None = None
processed_file_path: str | None = None

# =========================
# 📤 Upload an image file
# =========================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global original_file_path, processed_file_path
    # Clear the upload directory
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARN] File not deleted: {e}")

    original_file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(original_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Enable preview immediately
    processed_file_path = original_file_path

    return {"filename": file.filename}

# =========================
# ♻️ Reset processing to the original image
# =========================
@app.post("/api/reset")
def reset_processing():
    global processed_file_path

    if original_file_path is not None and os.path.exists(original_file_path):
        processed_file_path = original_file_path
        return {"filename": os.path.basename(original_file_path)}

    return {"error": "No original image available."}

# =========================
# 📥 Download the current processed image or PDF
# =========================
@app.get("/api/download/{filename}")
def download_file(filename: str):
    global processed_file_path

    if processed_file_path is None or not os.path.exists(processed_file_path):
        return {"error": "No processed file available for download."}

    return FileResponse(
        path=processed_file_path,
        filename=os.path.basename(processed_file_path),
        media_type="application/octet-stream"
    )

# =========================
# 📄 Preview endpoint (used in <iframe>, <img>, etc.)
# =========================
@app.get("/api/preview/{filename}")
def preview_file(filename: str):
    global processed_file_path

    if processed_file_path is None or not os.path.exists(processed_file_path):
        return {"error": "No file available for preview."}

    mime_type, _ = mimetypes.guess_type(processed_file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Force inline rendering (e.g. show PDF/image in browser)
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"'
    }

    return FileResponse(
        path=processed_file_path,
        filename=filename,
        media_type=mime_type,
        headers=headers
    )

# =========================
# 🎯 Character Detection step
# =========================
@app.post("/api/detect")
async def detect_characters():
    global original_file_path, processed_file_path
    if original_file_path is None or not os.path.exists(original_file_path):
        raise HTTPException(status_code=400, detail="No file uploaded.")

    input_path = processed_file_path or original_file_path
    filename = os.path.basename(input_path)

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a PDF file.")

    # returns a result for UI Presentation
    source_path = character_detection.run(input_path)

    filename = os.path.basename(source_path)
    output_filename = f"character_detection_{filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    # Copy the result to the desired output path
    shutil.copyfile(source_path, output_path)

    processed_file_path = output_path

    return {
        "filename": os.path.basename(processed_file_path)
    }

# =========================
# ✂️ Character Segmentation/Isolation step
# =========================
@app.post("/api/segment")
async def segment_characters():
    global original_file_path, processed_file_path

    if original_file_path is None or not os.path.exists(original_file_path):
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # returns a result for UI Presentation
    source_path = character_segmentation.run()

    filename = os.path.basename(source_path)
    output_filename = f"character_segmentation_{filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    # Copy the result to the desired output path
    shutil.copyfile(source_path, output_path)

    processed_file_path = output_path

    return {
        "filename": os.path.basename(processed_file_path)
    }

# =========================
# 🧠 Character Recognition (OCR) step
# =========================
@app.post("/api/recognize")
async def recognize_characters():
    global original_file_path, processed_file_path

    if original_file_path is None or not os.path.exists(original_file_path):
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # returns a result for UI Presentation
    source_path = character_recognition.run()

    filename = os.path.basename(source_path)
    output_filename = f"character_recognition_{filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    # Copy the result to the desired output path
    shutil.copyfile(source_path, output_path)

    processed_file_path = output_path

    return {
        "filename": os.path.basename(processed_file_path)
    }

# =========================
# 🧠 Word Recognition (LLM) step
# =========================
@app.post("/api/recognize_words")
async def recognize_words():
    global original_file_path, processed_file_path

    if original_file_path is None or not os.path.exists(original_file_path):
       raise HTTPException(status_code=400, detail="No file uploaded.")

    # returns a result for UI Presentation
    source_path = word_recognition.run()

    filename = os.path.basename(source_path)
    output_filename = f"word_recognition_{filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    # Copy the result to the desired output path
    shutil.copyfile(source_path, output_path)

    processed_file_path = output_path

    return {
        "filename": os.path.basename(processed_file_path)
    }


# =========================
# 📄 Create readable PDF
# =========================
@app.post("/api/create-pdf")
async def create_pdf():
    global original_file_path, processed_file_path

    if original_file_path is None or not os.path.exists(original_file_path):
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # returns a result for UI Presentation
    processed_pdf_path = pdf_creation.run(UPLOAD_DIR)

    processed_file_path = processed_pdf_path

    return {
        "filename": os.path.basename(processed_file_path)
    }