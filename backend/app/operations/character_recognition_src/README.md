# Character Recognition (character_recognition)

This folder contains the character recognition stage of the OCR pipeline. It provides scripts and utilities to prepare datasets, run text recognition models, and produce JSON and image outputs consumed by the backend and frontend UI.

## Purpose

- Convert a folder of OCR images into recognized text and produce structured JSON outputs.
- Provide a CLI-driven entrypoint suitable for being invoked by the backend operation wrapper (subprocess) or run manually during development.
- Keep model and I/O wiring concentrated here so the API wrapper only needs to provide input/output paths (no config file dependency by default).

## Key files

- `character_recognition_main.py` — Main CLI entrypoint. Accepts positional arguments instead of reading `config.json`.
- `utilities.py` — Helper functions used by the recognition script (image listing, aggregation, JSON writeout, etc.).
- `dataset_preparation.py` — Tools and helpers to build datasets or convert labeling formats for training/experimentation.
- `textRecognition.py` — Model inference and recognition utilities (model-specific code). 
- `installRequirements.py` — Script to install required Python packages (used optionally by `character_recognition_main.py` during startup).
- `requirements.txt` — Requirement pins specific to this module (may be used with pip).
- `data/` — Example or runtime data subfolder (e.g., `data/output/` for generated artifacts).

## CLI Usage

The preferred way to run recognition for development or when invoked by the backend is via the CLI entrypoint `character_recognition_main.py`.

Positional arguments expected by `character_recognition_main.py`:

1. `input_dir` (required) — path to a folder containing input images to process.
2. `save_img_path` (required) — directory where annotated images or output artifacts will be written.
3. `model_name` (optional) — model identifier to select which local model to use; defaults to `PP-OCRv5_server_rec`.

Example (PowerShell):

```powershell
# from repository root
cd E:\ocr_code\OCR_Chinese_Medicine_Project\backend\app\operations\character_recognition

# Use the active Python interpreter (recommended: the project's virtualenv)
python .\character_recognition_main.py \
  ..\..\uploaded_files\my_images \
  ..\..\processed_files\rec_output \
  ..\..\processed_files\rec_output\output_full.json \
  PP-OCRv5_server_rec
```

If you prefer to run the script using a specific virtual environment interpreter (for example when the backend operation wrapper does this), point the wrapper to the full `python.exe` path and pass the same arguments.

Example invoked by backend wrapper (Windows example):

```powershell
# Example interpreter used by operation wrapper
E:\ocr_code\OCR_Chinese_Medicine_Project\backend\.venv310\Scripts\python.exe \
  E:\ocr_code\OCR_Chinese_Medicine_Project\backend\app\operations\character_recognition\character_recognition_main.py \
  C:\path\to\input_images \
  C:\path\to\save_images \
  PP-OCRv5_server_rec
```

## Development / Setup

- The module can install its own runtime packages via `installRequirements.py`. For reproducible environments prefer creating and activating a virtual environment and installing `requirements.txt` with pip:

```powershell
cd E:\ocr_code\OCR_Chinese_Medicine_Project\backend\app\operations\character_recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Note about repository venv names: the repository contains both `venv` (used by `start_backend.bat`) and `.venv310` references in operation wrappers. Ensure the wrapper or your shell uses the same environment as the one with the required packages.

## Integration notes (backend API)

- The backend operation wrapper is expected to call this script via subprocess and pass the `input_dir` and `save_img_path` arguments. This README assumes the script will be invoked with CLI arguments (no `config.json` read).
- Two safe integration approaches:
  - Pass full CLI args from the wrapper (recommended) so the script is entirely driven by the API call.
  - Have the wrapper write a temporary config file and call the script (less preferred here because `character_recognition_main.py` is written to prefer CLI args).

## Outputs

- JSON summary: by default `output_full.json` inside the `save_img_path` folder. This file contains recognized text and per-image metadata that the frontend can preview.
- Annotated images (if produced) will be written under `save_img_path`.

## Troubleshooting

- "Module not found" or missing dependency errors: confirm you activated the correct venv and installed `requirements.txt` from this folder or the repo root `backend/requirements.txt`.
- If subprocess calls from the backend fail, check the wrapper's `env_python` path (commonly `backend\.venv310\Scripts\python.exe`) and adjust it or create the venv expected by the wrapper.
- If output JSON is not being created, check STDOUT/STDERR for the called process (the backend wrapper should capture and log these). Run the CLI manually to see stack traces.




