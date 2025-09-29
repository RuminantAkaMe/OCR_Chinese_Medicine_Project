# 🖼️ OCR Processing Webapp with Vue & FastAPI

A modern web service for uploading, processing, and downloading handwritten Chinese character manuscripts.  
The project uses a Vuetify frontend with Vue 3 & Pinia, and a FastAPI backend powered by Pillow for image handling.
Multiple AI approaches are used to detect, segment and recognize single characters and even whole words.

---

## 🚀 Features

- 📤 Upload images via the frontend
- ✨ Step-by-step image processing pipeline:
  - Character Detection
  - Character Segmentation
  - Character Recognition (OCR)
  - Word Recognition
  - PDF Creation
- 📥 Live preview and instant download of processed files
- 🔄 Reset functionality to restore the original file
- 🖼 Real-time image preview during processing
- 🔔 Snackbar feedback after every action

🚀 After setup, you should be able to start the application by using "start_all.batch" (on Windows)

---

## 🛠️ Technologies

- Frontend: Vue 3, TypeScript, Pinia, Vuetify, Vite
- Backend: FastAPI, Python, Pillow (PIL)
- Other Tools: Git, Node.js, pip, virtualenv
- AI Tools: Pytorch, CUDA, LLava VLM, YOLO, ...

---

## 📂 Project Structure

```plaintext
OCR_Chinese_Medicine_Project
├── frontend              # Vue + Vuetify application
│   └── src               # Components, Views, Store
├── backend               # FastAPI backend
│   ├── app
│   │   ├── main.py        # API routes (upload, processing, reset, download)
│   │   ├── operations/    # Separate Python files for each processing step
│   └── uploaded_files     # Temporary storage for uploaded/processed files
└── README.md              # This document

```

## 🛠️ Setup

⚙️ Setup-Instructions (In a git bash)

📦 Requirements
--------------------------------------------
Node.js + npm
Python 3.10+
Git
--------------------------------------------

🖼 start frontend
--------------------------------------------
```plaintext
cd frontend
npm install
npm run dev
```
--------------------------------------------

🔧 Setup backend
  Notice: Individual evironments are used for each processing stage!
--------------------------------------------
```plaintext
cd backend
python -m venv venv
venv/Scripts/activate           # Windows
# or: source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```
--------------------------------------------



🔐 .gitignore advice
.gitignore should make sure that these folders arent tracked:
--------------------------------------------
gitignore
```plaintext
# Backend
.venv310/
venv/
__pycache__/
uploaded_files/
backend/app/operations/word_recognition_src/data/
checkpoints_cv/
checkpoints/
data/
checkpoint_inference/

# Frontend
node_modules/
dist/

# Allgemein
*.log
.DS_Store
```
--------------------------------------------

## 🛠️ General Tips:

# Character Detection:

  For additional Info, see the individual files documentation.

# Character Segmentation:

  For additional Info, see the individual files documentation.

# Character Recognition:

  For additional Info, see the individual files documentation.

# Word Recognition:
  For Inference, you have to have your llava-onevision-qwen2-0.5b-ov-hf model in PATH,
  and change the MODEL_PATH in inference_llava.py to this PATH.
  Also, make use of checkpoints, you have to store the data of a checkpoint 
  in a ../operations/word_recognition_src/checkpoint_inference folder

  For additional Info, see the individual files documentation.