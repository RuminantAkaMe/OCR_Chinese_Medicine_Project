# Chinese Character Detection from Historical Documents

## 🧾 Project Overview
This project detects **Chinese characters from medieval paper images (PDF or image format)** using **YOLOv10**.  
The system performs the following steps automatically:

- Converts PDF pages into images  
- Removes red annotation marks  
- Detects individual Chinese characters  
- Groups them into **vertical text lines** (traditional layout)

---

## 🧑‍💻 Team Member
**Character Detection Module**

---

## 🧩 Prerequisites

- Python 3.10 or higher  
- Poppler (required by `pdf2image` if working with PDFs)
- Git (for cloning the repository)

---

## ⚙️ Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd OCR_Chinese_Medicine_Project-main/backend
```

---

### Step 2: Install Poppler
**Windows**
```bash
Download from: https://github.com/oschwartz10612/poppler-windows/releases
Extract and add the bin folder to your PATH
```

**Mac**
```bash
brew install poppler
```

**Linux**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

---

### Step 3: Create a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv310

# Activate (Mac/Linux):
source .venv310/bin/activate

# Activate (Windows):
.venv310\Scripts\activate
```

---

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install fastapi uvicorn opencv-python numpy ultralytics pdf2image pillow python-multipart pytest pytest-cov reportlab
```

---

## 🗂️ Project Structure

```
OCR_Chinese_Medicine_Project-main/
├── backend/
│   ├── app/
│   │   └── operations/
│   │       ├── character_detection.py (API wrapper)
│   │       └── character_detection_src/
│   │           ├── character_detection.py (Main detection script)
│   │           ├── evaluate.py
│   │           ├── train.py
│   │           ├── best.pt (Trained model)
│   │           ├── data.yaml
│   │           ├── yolov10n.pt
│   │           ├── LabeledImage/
│   │           │   ├── 7.jpg
│   │           │   ├── 7.jpg.json
│   │           │   ├── 8_org.jpeg
│   │           │   └── 8_org.jpeg.json
│   │           └── test_data/ (for testing)
│   │               └── sample.pdf
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_character_detection.py
│   │   └── test_integration.py
│   ├── .venv310/
│   ├── requirements.txt
│   ├── verify_pipeline.py
│   ├── test_pipeline.py
│   └── create_test_pdf.py
└── README.md
```

| File/Folder | Description |
|--------------|-------------|
| `character_detection.py` (wrapper) | API interface that calls the main detection script via subprocess |
| `character_detection.py` (main) | Core detection logic (PDF/image → character bounding boxes) |
| `evaluate.py` | Evaluates model accuracy (Precision, Recall, F1, mAP, IoU) |
| `train.py` | Script to train the YOLOv10 model on a custom dataset |
| `best.pt` | Trained YOLOv10 model weights |
| `data.yaml` | Configuration file defining dataset paths for training/validation |
| `yolov10n.pt` | Base YOLOv10n model (optional or pretrained) |
| `LabeledImage/` | Folder containing labeled data or test samples |
| `verify_pipeline.py` | Verification script to check if setup is correct |
| `test_pipeline.py` | End-to-end testing script |
| `create_test_pdf.py` | Utility to generate test PDF files |

---

## 🧪 Testing Your Branch Locally

### Quick Verification Guide

Follow these steps to ensure your character detection branch is working correctly:

#### Step 1: Verify Setup
First, check if everything is installed correctly:

```bash
cd backend
python verify_pipeline.py
```

**Expected Output:**
```
🔍 Verifying Character Detection Pipeline

✅ Wrapper found
✅ Actual script found
✅ Virtual environment found
✅ YOLO model found

📦 Checking dependencies:
✅ OpenCV (cv2) installed
✅ NumPy installed
✅ Ultralytics (YOLO) installed
✅ pdf2image installed

==================================================
Pipeline Flow:
==================================================
1. API receives PDF upload
2. Calls: operations/character_detection.py
3. Which runs: subprocess with python
4. Which executes: character_detection_src/character_detection.py
5. Results saved in: character_detection_src/data/output/
6. Returns: path to page_1.jpg
==================================================
```

---

#### Step 2: Create Test PDF
Generate a simple test PDF for testing:

```bash
python create_test_pdf.py
```

**Output:**
```
✅ Test PDF created at: app/operations/character_detection_src/test_data/sample.pdf
```

---

#### Step 3: Run Complete Pipeline Test
Test the entire character detection pipeline:

```bash
python test_pipeline.py
```

**Expected Output:**
```
🔍 Testing with PDF: .../test_data/sample.pdf
🚀 Running character detection pipeline...

[INFO] Converting PDF to images...
[INFO] Running character detection...
[INFO] Saved detection result for page_0.jpg
[INFO] Saved detection result for page_1.jpg
[INFO] Process completed.

✅ SUCCESS!
📄 Result saved at: .../data/output/page_1.jpg
✅ Output file verified: page_1.jpg

📊 Generated 2 image(s):
   • page_0.jpg
   • page_1.jpg
```

---

#### Step 4: Run Unit Tests (Optional)
Run the complete test suite using pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app/operations/character_detection_src --cov-report=html

# Run specific test file
pytest tests/test_character_detection.py
```

**Expected Output:**
```
============================= test session starts ==============================
collected 15 items

tests/test_character_detection.py ........                              [ 53%]
tests/test_integration.py .......                                       [100%]

============================== 15 passed in 2.34s ===============================
```

---

#### Step 5: Test with Your Own PDF
To test with your own PDF file:

```bash
cd app/operations/character_detection_src
python character_detection.py "path/to/your/test.pdf" "data/output"
```

**Example:**
```bash
python character_detection.py "/Users/username/Documents/my_document.pdf" "data/output"
```

---

### Troubleshooting Tests

| Issue | Solution |
|-------|----------|
| **"Test PDF not found"** | Run `python create_test_pdf.py` first |
| **"Module not found"** | Make sure virtual environment is activated and dependencies installed |
| **"Poppler not found"** | Install poppler (see installation steps above) |
| **"best.pt not found"** | Ensure your trained model is in `character_detection_src/best.pt` |
| **Tests fail on Mac/Linux** | Check that `operations/character_detection.py` uses `sys.executable` for Python path |

---

## 🚀 Usage in Production

### 1. Run Character Detection (Direct Script)
```bash
cd app/operations/character_detection_src
python character_detection.py "input.pdf" "data/output"
```

**What it does:**
- Converts PDF pages to images (300 DPI)
- Removes red marks (if present)
- Detects and groups Chinese characters
- Saves results with colored bounding boxes

---

### 2. Run via API Wrapper
```python
from app.operations import character_detection

pdf_path = "path/to/your/document.pdf"
result_path = character_detection.run(pdf_path)
print(f"Result saved at: {result_path}")
```

---

### 3. Evaluate Model Performance
```bash
cd app/operations/character_detection_src
python evaluate.py
```

**Outputs:**
- Precision, Recall, F1-Score, and mAP  
- Mean IoU for predicted boxes  
- Precision-Recall curves and metric plots  

> Results and visualizations are automatically saved in the output directory.

---

### 4. Train Model on Custom Dataset
If you want to train the YOLOv10 model on your own dataset:

```bash
cd app/operations/character_detection_src
python train.py
```

**How to set up your custom dataset:**
1. Open the `data.yaml` file.  
2. Modify the paths for your **training** and **validation** datasets.  
   Example:
   ```yaml
   train: path/to/your/train/images
   val: path/to/your/val/images
   nc: 1
   names: ['character']
   ```
3. Create separate folders for training and validation:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```
4. Place your labeled images and corresponding YOLO-format text files inside these folders.  
5. Run `train.py` to start training — a new `best.pt` will be generated upon completion.

---

> ⚠️ **IMPORTANT WARNING — MODEL PATH CHANGE REQUIRED**
>
> When you train your own dataset using `train.py`, YOLO will automatically save your trained model at:  
> **`chinese_characters_model_yolo10/weights/best.pt`**
>
> Therefore, if you plan to use this newly trained model in `character_detection.py`,  
> you **must update** the following line inside that file:
>
> ```python
> trained_model_path = 'best.pt'
> ```
>
> Change it to:
>
> ```python
> trained_model_path = 'chinese_characters_model_yolo10/weights/best.pt'
> ```
>
> This ensures that the detection script uses your latest trained model instead of the default one.

---

## ⚙️ Configuration Parameters

### Inside `character_detection.py`
```python
dpi = 300              # PDF to image resolution
conf_threshold = 0.3   # YOLO detection confidence
x_threshold = 50       # Distance threshold for grouping vertical text lines
```

### Inside `evaluate.py`
```python
iou_threshold = 0.1
conf_threshold = 0.5
plot_pr_curve = True
analyze_thresholds = True
```

---

## 🧠 Model Information

| Property | Details |
|-----------|----------|
| Model | YOLOv10n (nano) |
| Task | Chinese character detection |
| Input Size | 640×640 |
| Model File | `best.pt` |
| Framework | PyTorch (`ultralytics`) |

---

## 💡 Performance Tips

- Use **lower confidence (0.2–0.3)** for high recall  
- Use **0.4–0.5** for balanced results  
- Use **0.6+** for high precision  
- For best OCR extraction, prioritize **recall** (capture all characters)

---

## 🧰 General Troubleshooting

| Issue | Solution |
|--------|-----------|
| **Poppler not found** | Install and add to PATH (`pdftoppm -h` should work) |
| **Model not found** | Check if `best.pt` is present in `character_detection_src/` folder |
| **CUDA errors** | Ensure GPU is available or run on CPU |
| **Low accuracy** | Lower confidence threshold or retrain model |
| **Subprocess errors** | Ensure `operations/character_detection.py` uses `sys.executable` |
| **Import errors** | Activate virtual environment: `source .venv310/bin/activate` |

---

## 📦 Dependencies

Core dependencies:
- `ultralytics` (YOLOv10)
- `torch`
- `opencv-python`
- `pdf2image`
- `matplotlib`
- `numpy`
- `fastapi`
- `uvicorn`
- `pillow`

Testing dependencies:
- `pytest`
- `pytest-cov`
- `pytest-mock`
- `reportlab`

Install all via:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install ultralytics torch opencv-python pdf2image matplotlib numpy fastapi uvicorn pillow pytest pytest-cov reportlab
```

---

## 🌈 Key Features

- Automatic PDF → Image conversion  
- Red mark removal (HSV + inpainting)  
- Vertical text line grouping  
- Color-coded bounding boxes  
- Comprehensive test suite  
- Evaluation with multiple metrics  
- Custom model training support via `train.py`  
- Lightweight YOLOv10n model  
- API wrapper for easy integration
- Cross-platform support (Windows, Mac, Linux)

---

## 🔄 Development Workflow

### Making Changes to Character Detection

1. **Make your changes** in `character_detection_src/character_detection.py`
2. **Test your changes:**
   ```bash
   # Quick test with direct script
   cd app/operations/character_detection_src
   python character_detection.py "test_data/sample.pdf" "data/output"
   
   # Test through wrapper
   cd backend
   python test_pipeline.py
   
   # Run unit tests
   pytest
   ```
3. **Commit and push** your changes
4. **Create a pull request** with test results

### Before Pushing Your Branch

Run this checklist:
```bash
# 1. Verify setup
python verify_pipeline.py

# 2. Run pipeline test
python test_pipeline.py

# 3. Run unit tests
pytest -v

# 4. Check code quality (optional)
pylint app/operations/character_detection_src/character_detection.py
```

All tests should pass before pushing! ✅

---

## 📬 Contact

For questions or issues, contact **hardik7393@gmail.com**.

---

## 🪪 License
Educational project for university coursework.

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | October 2025 | Initial release |
| 1.1 | October 2025 | Added testing framework and documentation |

**Last Updated:** October 2025  
**Current Version:** 1.1
