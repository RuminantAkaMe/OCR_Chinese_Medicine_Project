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

- Python 3.8 or higher  
- Poppler (required by `pdf2image` if working with PDFs)

---

## ⚙️ Installation

### Step 1: Install Poppler
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

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

**Activate (Mac/Linux):**
```bash
source venv/bin/activate
```

**Activate (Windows):**
```bash
venv\Scripts\activate
```

---

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
project_folder/
├── README.md
├── character_detection.py
├── evaluate.py
├── train.py
├── best.pt
├── data.yaml
├── yolov10n.pt
├── LabeledImage/
    └── 7.jpg (1st image for testing purposes)
    └── 7.jpg.json (contains the coordinates for the manually labelled file for the 1st image)
    └── 8_org.jpeg (2nd image for testing purposes)
    └── 8_org.jpeg.json (contains the coordinates for the manually labelled file for the 2nd image)
└── Datasets/  (Optional)
    └── 1800.pdf
```

| File/Folder | Description |
|--------------|-------------|
| `character_detection.py` | Main script for detection (PDF/image input → character bounding boxes) |
| `evaluate.py` | Evaluates model accuracy (Precision, Recall, F1, mAP, IoU) |
| `train.py` | Script to train the YOLOv10 model on a custom dataset |
| `best.pt` | Trained YOLOv10 model weights |
| `data.yaml` | Configuration file defining dataset paths for training/validation |
| `yolov10n.pt` | Base YOLOv10n model (optional or pretrained) |
| `LabeledImage/` | Folder containing labeled data or test samples, used in the evaluate.py file |
| `Datasets/1800.pdf` | Input PDF file for detection if user hasn’t provided their own |

---

## 🚀 Usage

### ⚠️ Before Running (if you want to run without providing your own PDF)
If you want to run the detection system:  
1. **Create a folder** named `Datasets` in the project root (if it doesn’t exist).  
2. **Upload your input PDF** inside the `Datasets` folder.  
3. **Rename the PDF file to** `1800.pdf`.  

Example:
```
project_folder/
└── Datasets/
    └── 1800.pdf
```

---

### 1. Run Character Detection
```bash
python character_detection.py
```

**What it does:**
- Converts PDF pages to images (300 DPI)
- Removes red marks (if present)
- Detects and groups Chinese characters
- Saves results with colored bounding boxes

> Input file path is fixed as `Datasets/1800.pdf`.

---

### 2. Evaluate Model Performance
```bash
python evaluate.py
```

**Outputs:**
- Precision, Recall, F1-Score, and mAP  
- Mean IoU for predicted boxes  
- Precision-Recall curves and metric plots  

> Results and visualizations are automatically saved in the output directory (defined in `evaluate.py`).

---

### 3. Train Model on Custom Dataset
If you want to train the YOLOv10 model on your own dataset, use `train.py`.  

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
dpi = 300          # PDF to image resolution
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

## 🧰 Troubleshooting

| Issue | Solution |
|--------|-----------|
| **Poppler not found** | Install and add to PATH (`pdftoppm -h` should work) |
| **Model not found** | Check if `best.pt` is present in root folder |
| **CUDA errors** | Ensure GPU is available or run on CPU |
| **Low accuracy** | Lower confidence threshold or retrain model |

---

## 📦 Dependencies

- `ultralytics` (YOLOv10)
- `torch`
- `opencv-python`
- `pdf2image`
- `matplotlib`
- `numpy`

Install all via:
```bash
pip install ultralytics torch opencv-python pdf2image matplotlib numpy
```

---

## 🌈 Key Features

- Automatic PDF → Image conversion  
- Red mark removal (HSV + inpainting)  
- Vertical text line grouping  
- Color-coded bounding boxes  
- Evaluation with multiple metrics  
- Custom model training support via `train.py`  
- Lightweight YOLOv10n model  

---

## 📬 Contact

For questions or issues, contact **hardik7393@gmail.com**.

---

## 🪪 License
Educational project for university coursework.

**Last Updated:** October 2025  
**Version:** 1.0
