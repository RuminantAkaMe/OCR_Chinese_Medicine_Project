# 🖼️ OCR Processing Webapp with Vue & FastAPI

A modern web service for uploading, processing, and downloading handwritten Chinese character images.  
The project uses a Vuetify frontend with Vue 3 & Pinia, and a FastAPI backend powered by Pillow for image handling.

---

## 🚀 Features

- 📤 Upload images via the frontend
- ✨ Step-by-step image processing pipeline:
  - Preprocessing
  - Character Detection
  - Character Segmentation
  - Character Recognition (OCR)
  - PDF Creation
- 📥 Live preview and instant download of processed files
- 🔄 Reset functionality to restore the original file
- 🖼 Real-time image preview during processing
- 🔔 Snackbar feedback after every action

---

## 🛠️ Technologies

- Frontend: Vue 3, TypeScript, Pinia, Vuetify, Vite
- Backend: FastAPI, Python, Pillow (PIL)
- Other Tools: Git, Node.js, pip, virtualenv

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
