# Chinese Character Isolation Module 🧾

## 📘 Project Overview
This module isolates and preprocesses **individual Chinese characters** from ancient handwritten documents.  
It is designed as **the second stage** after character detection (YOLO-based) in the **BIMAP OCR Pipeline**.

The system performs the following steps automatically:
- Reads bounding box JSONs (from detection stage)
- Crops character regions from page images
- Performs noise removal, binarization, and size normalization
- Exports clean **64×64 character patches** for recognition and semantic annotation

---

## 🧩 Prerequisites
- Python 3.8 or higher  
- Pillow, OpenCV, NumPy, Pandas  

---

### Step 1️⃣ Clone or Navigate to the Repository
```bash
git clone https://github.com/weiweizhang/OCR_Chinese_Medicine_Project.git
cd backend/app/operations/character_isolation
```

---

### Step 2: Create a Virtual Environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows


---

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
character_isolation/
├── README.md
├── character_isolation.py
├── unified_preprocess.py
├── eval_preprocess_quality.py
├── eval_normalization.py
├── processing_steps.png
└── requirements.txt
```

| File                         | Description                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| `character_isolation.py`     | Crop characters from page images based on YOLO bounding boxes |
| `unified_preprocess.py`      | Unified denoising, contrast enhancement, normalization        |
| `eval_preprocess_quality.py` | Evaluate preprocessing effect for 4 quality categories        |
| `eval_normalization.py`      | Quantitative evaluation of normalized character consistency   |
| `processing_steps.png`       | Visualization of preprocessing pipeline                       |
| `requirements.txt`           | Dependencies list                                             |


---

## 📊 Evaluation Metrics

Structural Similarity Index (SSIM)

Laplacian Variance / Edge Density

Character Occupancy Ratio

Recognition Accuracy Gain

---

---

## 📦 Dependencies

matplotlib==3.10.6
numpy==2.3.3
pdf2image==1.17.0
torch==2.8.0
torchvision==0.23.0
ultralytics==8.3.203
Pillow==9.5.0
opencv-python>=4.8.0

---


## 📬 Contact
📧 weiwei19980422@gmail.com
---

## 🪪 License
Reserach project in AIBE(FAU)

**Last Updated:** October 2025  
**Version:** 1.0
