# 🧩 Character Isolation Pipeline for Chinese Veterinary OCR

This repository contains the **character isolation and preprocessing pipeline** developed as part of the *BIMAP Project – OCR of Chinese Veterinary Texts* at **Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)**.

The goal of this module is to isolate, clean, and normalize individual Chinese characters from scanned historical manuscripts.  
It refines bounding box detections from the character detection module and outputs standardized **64×64 pixel** character images for downstream OCR training.

---

## 🧠 Research Motivation

Optical Character Recognition (OCR) on historical Chinese medical texts faces challenges such as:
- Low contrast and noise due to paper aging or ink degradation.
- Touching or overlapping characters.
- Variations in handwriting and irregular layouts.

This preprocessing pipeline addresses these issues by:
1. Normalizing image size and brightness.  
2. Filtering out invalid crops and artefacts.  
3. Improving character continuity and consistency.  

The processed outputs are designed to improve **OCR recognition accuracy** and facilitate **dataset annotation**.

---

## ⚙️ Technical Overview

### 🔄 Pipeline Stages

1. **Input Parsing**
   - Read bounding box coordinates from JSON files (output of YOLO-based detection).
   - Match them with corresponding scanned page images.

2. **Cropping & Cleaning**
   - Crop character regions using Pillow (PIL).  
   - Convert to grayscale and remove small noisy patches.

3. **Normalization**
   - Resize valid characters to **64×64 px** using bilinear interpolation.  
   - Center and pad strokes for uniform appearance.

4. **Filtering**
   - Remove empty, blurry, or invalid crops based on pixel intensity thresholds.

5. **Evaluation**
   - Compute SSIM, Precision, Recall, and F1 to assess image quality.

---

## 🧩 Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RuminantAkaMe/OCR_Chinese_Medicine_Project.git
cd OCR_Chinese_Medicine_Project/backend/app

2️⃣ Create a Python Environment

Use Conda (recommended):

conda create -n ocr_env python=3.10
conda activate ocr_env

3️⃣ Install Dependencies
pip install -r requirements.txt

For this submodule (character isolation only):

cd operations
pip install pillow opencv-python numpy matplotlib

Prepare Input Data

Create the following folders and add your input data:

backend/app/
 ├── json_boxes/       # bounding box annotations (.json)
 │   ├── page_001.json
 │   ├── page_002.json
 ├── raw_images/       # scanned pages (.jpg/.png)
 │   ├── page_001.png
 │   ├── page_002.png


Each JSON file should correspond to a document image with the same name.

5️⃣ Run the Isolation Script

To isolate, clean, and normalize all characters:

python character_isolation.py


This will:

Read bounding boxes from JSON

Crop each character

Clean and resize to 64×64 px

Save all valid outputs to results/

6️⃣ Visualize the Processing Steps

To visualize the image at each stage:

python character_isolation.py --show


Example output:

Step	Visualization
Raw Image	

Grayscale	

Binarized	

After Cleaning	

Final Normalized	
7️⃣ Check Results

Output images are stored in:

backend/app/results/


Each image corresponds to a detected character, normalized to 64×64 pixels and named according to its source page and index.

📊 Example Comparison
Input (Bounding Boxes)	Output (Normalized 64×64)

	

(You can replace the example images in backend/app/Figures/ with your own.)

🧠 Evaluation Metrics

To evaluate the quality of isolated characters, the following metrics are used:
SSIM(x,y) Precision ,Recall​,F1
	​
	​

🧩 Integration with OCR System

This module is designed to integrate with the overall OCR workflow:

Character Detection → YOLO-based bounding box extraction
Character Isolation (this module) → Cropping, cleaning, normalization
Character Recognition → CNN-based classification
Text Reconstruction → Line-level ordering and merging

📜 License

This project is released under the MIT License.
See the LICENSE
 file for details.

👤 Author

Weiwei Zhang
Department of Medical Engineering
Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)
📧 weiwei19980422@gmail.com
