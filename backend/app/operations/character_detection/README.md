
Markdown

# Chinese Character Detection from Medieval Documents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project detects Chinese characters from medieval paper images (in PDF format) using YOLOv10. The system converts PDF pages to images and identifies individual characters with bounding boxes, which are then grouped into vertical text lines.

---

## 🚀 Project Overview

This project provides a complete pipeline for processing historical documents:
1.  **Preprocessing**: Automatically converts PDFs to high-quality images and removes common visual noise like red annotation marks.
2.  **Detection**: Utilizes a trained YOLOv10 model to accurately locate Chinese characters.
3.  **Analysis**: Groups detected characters into vertical columns, reflecting traditional Chinese text layout, and color-codes them for easy visualization.
4.  **Evaluation**: Includes robust scripts to measure model performance with metrics like mAP, Precision, and Recall.

---

## 🛠️ Prerequisites

-   Python 3.8 or higher
-   Poppler (for PDF to image conversion)

---

## ⚙️ Installation

### Step 1: Install Poppler

**On Windows:**
1.  Download the latest release from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
2.  Extract the ZIP file to a location like `C:\poppler`.
3.  Add the extracted `bin` folder (e.g., `C:\poppler\bin`) to your system's PATH environment variable.

**On macOS:**
'''bash
brew install poppler
On Linux (Ubuntu/Debian):

Bash

sudo apt-get update
sudo apt-get install poppler-utils
Step 2: Create a Virtual Environment (Recommended)
Bash

# Create the virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
Step 3: Install Python Dependencies
Bash

pip install -r requirements.txt
📁 Project Structure
.
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── predict.py             # Main prediction script
├── evaluate.py            # Model evaluation with metrics
├── train.py               # Model training script
├── cleaned.py             # PDF preprocessing utility
├── data.yaml              # YOLO dataset configuration
├── yolo10n.yaml           # YOLO architecture configuration
├── models/                # Model weights directory
│   └── chinese_characters_model3/
│       └── weights/
│           └── best.pt    # Trained model weights
├── Datasets/              # Input data
│   ├── 1800.pdf           # Sample PDF
│   └── labeled/           # Ground truth annotations
│       ├── 8_org.jpeg
│       └── 8_org.jpeg.json
├── temp_images/           # Temporary converted images (auto-generated)
├── cleaned_images/        # Preprocessed images (auto-generated)
├── results_7/             # Detection results (auto-generated)
├── coords/                # Bounding box coordinates JSON (auto-generated)
└── Hardik_Final/          # Evaluation outputs (auto-generated)
▶️ Usage
1. Preprocess PDF (Remove Red Circle Marks)
This script is designed to remove red annotation marks often found in medieval documents before running detection.

Bash

python cleaned.py
Input: Datasets/1800.pdf

Output: Cleaned images in the cleaned_images/ folder.

How it works:

Converts PDF to high-quality images (300 DPI).

Detects red circles/marks using the HSV color space.

Removes them using OpenCV's inpainting technique.

Saves cleaned images ready for the detection step.

2. Run Character Detection
This is the main script to run inference on a PDF.

Bash

python predict.py
How it works:

Converts PDF pages to images and automatically removes red annotations.

Detects Chinese characters using the trained YOLO model.

Groups characters into vertical text lines based on their horizontal proximity.

Colors each vertical line differently for easy visualization.

Outputs:

results_7/: Images with colored detection boxes.

coords/: JSON files containing bounding box coordinates for each page.

3. Evaluate Model Performance
This script compares model predictions against ground truth data to calculate performance metrics.

Bash

python evaluate.py
What it does:

Calculates Precision, Recall, F1-Score, mAP, and Mean IoU.

Generates precision-recall curves to visualize performance.

Analyzes performance at different confidence thresholds (0.1 to 0.9) to help select the optimal value.

Outputs (in Hardik_Final/ folder):

precision_recall_curve.png: The overall PR curve.

threshold_analysis.png: Plots showing Precision/Recall/F1 vs. Confidence.

result_debug.jpg: A visual comparison with ground truth (blue), correct detections (green), and false positives (red).

4. Train a Custom Model (Optional)
Train the YOLOv10 model on your own dataset.

Bash

python train.py
Requires an annotated dataset defined in data.yaml.

Saves the newly trained model to the models/ directory.

🔧 Configuration Parameters
predict.py
Python

dpi = 300           # Image quality for PDF conversion
conf = 0.3          # Confidence threshold for detection
x_threshold = 50    # Pixel distance for grouping vertical lines
pad_pixels = 4      # Padding around bounding boxes
evaluate.py
Python

iou_threshold = 0.1   # IoU threshold for matching predictions
conf_threshold = 0.5  # Confidence threshold for evaluation
plot_pr_curve = True  # Generate precision-recall curve
train.py
Python

epochs = 15         # Number of training epochs
imgsz = 640         # Input image size
pretrained = False  # Train from scratch
🧠 Model Information
Architecture: YOLOv10n (nano version - lightweight and fast)

Task: Object detection (Chinese character bounding boxes)

Training: Trained on a custom dataset of medieval Chinese documents.

Input Size: 640x640 pixels

Model Path: models/chinese_characters_model3/weights/best.pt

✨ Key Features
PDF Processing: Automatic conversion of PDF documents to high-quality images.

Noise Removal: Intelligent removal of red annotation circles using HSV color detection.

Vertical Text Detection: Groups characters into traditional vertical text lines.

Color-Coded Visualization: Assigns each text column a unique color for easy distinction.

Comprehensive Evaluation: Provides multiple metrics and visualizations for model assessment.

JSON Output: Saves bounding box coordinates for further processing.

💡 Performance & Recommendations
For medieval Chinese characters, lower confidence thresholds (0.2-0.3) are recommended to maximize character recall.

0.2-0.3: High recall, catches most characters (recommended for OCR).

0.4-0.5: Balanced precision and recall.

0.6+: High precision, fewer false positives but may miss characters.

🐛 Troubleshooting
"Poppler not found" error: Ensure Poppler is installed and its bin directory is in your system's PATH.

"Model file not found" error: Verify the model path is correct: models/chinese_characters_model3/weights/best.pt.

Low detection accuracy: Try lowering the conf threshold in predict.py to 0.2 or 0.3.

Too many false positives: Increase the conf threshold to 0.5 or higher.

📜 License
This project is for educational purposes as part of university coursework.

📞 Contact & Support
For questions or issues, please create an issue in the project repository or contact the course instructor.
