# Chinese Character Detection from Historical Documents

## 🧾 Project Overview
This project detects **Chinese characters from medieval paper images (PDF or image format)** using **YOLOv10**.  
The system performs the following steps automatically:

- Converts PDF pages into images  
- Removes red annotation marks  
- Detects individual Chinese characters  
- Groups them into **vertical text lines** (traditional layout)

---

## 🧑‍💻 Module Name
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
git clone 'https://github.com/RuminantAkaMe/OCR_Chinese_Medicine_Project'
go to OCR_Chinese_Medicine_Project-main/backend/app/operations/character_detection_src/
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
python -m venv venv

# Activate (Mac/Linux):
source venv/bin/activate

# Activate (Windows):
venv\Scripts\activate
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
character_detection_branch/
├── LabeledImage/
│   ├── 7.jpg
│   ├── 7.jpg.json
│   ├── 8_org.jpeg
│   └── 8_org.jpeg.json
├── README.md
├── best.pt
├── character_detection.py
├── data.yaml
├── empty_placeholder_script.py
├── evaluate.py
├── requirements.txt
└── train.py
```

| File/Folder | Description |
|--------------|-------------|
| `character_detection.py` | Main script for detection (PDF/image input → character bounding boxes) |
| `evaluate.py` | Evaluates model accuracy (Precision, Recall, F1, mAP, IoU) |
| `train.py` | Script to train the YOLOv10 model on a custom dataset |
| `best.pt` | Trained YOLOv10 model weights |
| `data.yaml` | Configuration file defining dataset paths for training/validation |
| `empty_placeholder_script.py` | Placeholder script for UI integration |
| `requirements.txt` | Python package dependencies |
| `LabeledImage/` | Folder containing labeled test samples for evaluation |
| `LabeledImage/7.jpg` | First test image |
| `LabeledImage/7.jpg.json` | Ground truth annotations for first test image |
| `LabeledImage/8_org.jpeg` | Second test image |
| `LabeledImage/8_org.jpeg.json` | Ground truth annotations for second test image |

---

## 🧪 Testing Character Detection Branch Locally

### Quick Testing Guide

Follow these steps to ensure the character detection branch is working correctly:

#### Prerequisites for Testing
1. Make sure you have activated your virtual environment:
   ```bash
   # Mac/Linux:
   source venv/bin/activate
   
   # Windows:
   venv\Scripts\activate
   ```

2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify that `best.pt` (trained model) exists in the project root directory

---

#### Testing the Character Detection

**Step 1: Navigate to the project directory**
```bash
cd character_detection_branch
```

**Step 2: Prepare a test PDF**
- Place your test PDF file in the project directory, or
- Use any PDF file you want to test

**Step 3: Run the detection script**
```bash
python character_detection.py "pdf_name.pdf" "output_folder"
```

**Example:**
```bash
# Using a test PDF in the current directory
python character_detection.py "test_document.pdf" "results"

# Using absolute path
python character_detection.py "/Users/username/Documents/sample.pdf" "results"
```

---

#### Expected Output

When the script runs successfully, you should see:

```
[INFO] Converting PDF to images...
[INFO] Running character detection...
[INFO] Saved detection result for page_0.jpg
[INFO] Saved detection result for page_1.jpg
[INFO] Process completed. Results saved to results
```

---

#### Checking the Results

After running the script, check the output folder:

```bash
# View the generated files
ls results/

# Expected files:
# - page_0.jpg (processed page 1 with bounding boxes)
# - page_1.jpg (processed page 2 with bounding boxes)
# - coords/ (folder containing JSON files with coordinates)
```

**What you should see in the output images:**
- Original document pages converted to images
- Colored bounding boxes around detected Chinese characters
- Each vertical line of text has a different color

**Coordinate files (in `results/coords/`):**
- `page_0.json` - Contains bounding box coordinates for page 1
- `page_1.json` - Contains bounding box coordinates for page 2

---

#### Verifying Success

Check if my branch is working correctly if:
- ✅ The script runs without errors
- ✅ Output images are generated in the specified output folder
- ✅ Characters are detected and marked with colored boxes
- ✅ JSON coordinate files are created in the `coords/` subfolder
- ✅ Red marks (if any) are removed from the document

---

#### Testing Different Scenarios

**Test 1: Single page PDF**
```bash
python character_detection.py "single_page.pdf" "results"
# Should generate: page_0.jpg
```

**Test 2: Multi-page PDF**
```bash
python character_detection.py "multi_page.pdf" "results"
# Should generate: page_0.jpg, page_1.jpg, page_2.jpg, etc.
```

**Test 3: Document with red annotations**
```bash
python character_detection.py "annotated_doc.pdf" "results"
# Red marks should be automatically removed
```

---

### Troubleshooting Tests

| Issue | Solution |
|-------|----------|
| **"ModuleNotFoundError"** | Activate virtual environment and install dependencies |
| **"Poppler not found"** | Install poppler (see installation steps above) |
| **"FileNotFoundError: best.pt"** | Ensure your trained model `best.pt` is in the project root |
| **"No such file or directory"** | Check your PDF path is correct (use absolute path) |
| **"No characters detected"** | Lower confidence threshold in the script (default: 0.3) |
| **Empty output folder** | Check if PDF has text content and verify model loaded correctly |

---

## 🚀 Usage

### 1. Run Character Detection
```bash
python character_detection.py "input.pdf" "output_folder"
```

**What it does:**
- Converts PDF pages to images (300 DPI)
- Removes red marks (if present)
- Detects and groups Chinese characters
- Saves results with colored bounding boxes

**Example:**
```bash
python character_detection.py "document.pdf" "results"
```

---

### 2. Evaluate Model Performance
```bash
python evaluate.py
```

**Outputs:**
- Precision, Recall, F1-Score, and mAP  
- Mean IoU for predicted boxes  
- Precision-Recall curves and metric plots  

> Uses labeled images from `LabeledImage/` folder for evaluation.
> Results and visualizations are automatically saved in the output directory.

---

### 3. Train Model on Custom Dataset
If you want to train the YOLOv10 model on your own dataset:

```bash
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
| Model | YOLOv10n|
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
| **Model not found** | Check if `best.pt` is present in project root folder |
| **CUDA errors** | Ensure GPU is available or run on CPU |
| **Low accuracy** | Lower confidence threshold or retrain model |
| **Import errors** | Activate virtual environment: `source venv/bin/activate` |
| **PDF conversion fails** | Verify Poppler is installed correctly |

---

## 📦 Dependencies

Core dependencies:
- `ultralytics` (YOLOv10)
- `torch`
- `opencv-python`
- `pdf2image`
- `matplotlib`
- `numpy`

Install all via:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install ultralytics torch opencv-python pdf2image matplotlib numpy
```

---

## 🌈 Key Features

- Automatic PDF → Image conversion  
- Red mark removal (HSV + inpainting)  
- Vertical text line grouping  
- Color-coded bounding boxes  
- Evaluation with multiple metrics (using labeled test data)
- Custom model training support via `train.py`  
- Lightweight YOLOv10n model  
- JSON output with character coordinates

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
| 1.0 | August 2025 | Initial release |
| 1.1 | October 2025 | Added testing framework and documentation |

**Last Updated:** October 2025  
**Current Version:** 1.1
