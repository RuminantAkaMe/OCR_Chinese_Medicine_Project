Chinese Character Isolation Module
🧾 Project Overview

This module isolates and preprocesses individual Chinese characters from ancient handwriiten documents.
It is designed as the second stage after character detection (YOLO-based) in the BIMAP OCR Pipeline.

The system performs the following steps automatically:

Reads bounding box JSONs (from detection stage)

Crops character regions from page images

Performs noise removal, binarization, and size normalization

Exports clean 64×64 character patches for recognition and semantic annotation



🧩 Prerequisites

Python 3.8 or higher

Pillow, OpenCV, NumPy, Pandas

⚙️ Installation
Step 1: Clone or navigate to the repository
cd backend/app/operations/character_isolation

Step 2: Create a Virtual Environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

Step 3: Install Dependencies
pip install -r requirements.txt

🗂️ Project Structure
character_isolation/
├── README.md
├── character_isolation.py
├── unified_preprocess.py
├── eval_preprocess_quality.py
├── eval_normalization.py
├── processing_steps.png
└── requirements.txt

File	Description
character_isolation.py	Crops characters from page images based on YOLO bounding boxes
unified_preprocess.py	Applies unified preprocessing (denoising, contrast, normalization)
eval_preprocess_quality.py	Evaluates preprocessing effect using 4 data categories
eval_normalization.py	Measures pixel and geometric consistency after normalization
processing_steps.png	Visualization of preprocessing pipeline
requirements.txt	Dependency list for environment setup
🚀 Usage
1️⃣ Character Isolation
python character_isolation.py


Output:

Extracted raw crops → raw_chars/

Normalized 64×64 characters → isolated_chars/

2️⃣ Unified Preprocessing
python unified_preprocess.py


Includes:

Adaptive thresholding

Morphological cleaning

Edge and contrast enhancement

Character size standardization

3️⃣ Evaluation
python eval_preprocess_quality.py


Outputs:

results_eval.csv – per-sample results

results_eval_summary.csv – averaged results per dataset

⚙️ Configuration Parameters

Inside character_isolation.py

COORDS_DIR = 'coords'          # JSON bounding box directory
IMAGES_DIR = 'temp_images'     # Source page images
RAW_DIR = 'raw_chars'          # Output raw crops
OUT_DIR = 'isolated_chars'     # Output cleaned 64×64 characters
TARGET_SIZE = (64, 64)         # Normalized image size

🌈 Example Visualization

Step	Description
Raw Image	Original crop from scanned document
Grayscale	Converted to grayscale
Binarized	Adaptive threshold for text–background separation
After Cleaning	Morphological filtering to remove noise
Final Normalized	Standardized 64×64 clean output
📊 Evaluation Metrics

SSIM (Structural Similarity)

Edge Density / Laplacian Variance

Contrast / Occupancy Ratio

Recognition Accuracy Gain

📦 Dependencies
matplotlib==3.10.6
numpy==2.3.3
pdf2image==1.17.0
torch==2.8.0
torchvision==0.23.0
ultralytics==8.3.203
Pillow==9.5.0
opencv-python>=4.8.0

🧰 Notes

Works seamlessly with YOLOv10-based detection results (page_*.json).

Each processed character can be directly used for OCR recognition or annotation.

Modularized for integration into backend/app/operations.

📬 Contact

For questions or issues, contact:
Weiwei Zhang – BIMAP P1 Project
📧 weiwei19980422@gamil.com

🪪 License

Research use only – educational purpose (BIMAP Project P1).
Last Updated: October 2025
Version: 1.0
