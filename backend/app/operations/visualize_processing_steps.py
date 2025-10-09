
"""
visualize_processing_steps.py
Visualize the main preprocessing stages:
Raw image → Grayscale → Binarized → Cleaned → Normalized (64×64)
"""

import os
import cv2
import matplotlib.pyplot as plt
from unified_preprocess import process_crop, sauvola_bin, light_close, to_gray

# === choose one normal sample from raw_chars ===
BASE = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE, "raw_chars", "page_3_char_011.png")  
print("📁 Using sample:", img_path)
if not os.path.exists(img_path):
    raise FileNotFoundError("❌ Cannot find this image, please check the file name.")

# Step 1: load raw image
img = cv2.imread(img_path)
raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: grayscale
gray = to_gray(img)

# Step 3: binarization (Sauvola)
bin_img = sauvola_bin(gray)

# Step 4: morphological cleaning (closing)
clean = light_close(bin_img)

# Step 5: normalization (resize to 64×64)
res = process_crop(img)
final = res["parts"][0] if res["parts"] else clean

# === plot all steps ===
plt.figure(figsize=(14, 3))
steps = [raw, gray, bin_img, clean, final]
titles = ["Raw Image", "Grayscale", "Binarized", "After Cleaning", "Final Normalized"]

for i, (im, title) in enumerate(zip(steps, titles), start=1):
    plt.subplot(1, 5, i)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")

# === save final normalized character ===
output_path = "final_normalized_char.png"  
cv2.imwrite(output_path, final)
print(f"✅ Final normalized image saved to: {output_path}")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/processing_steps_normal.png", dpi=300)
print("✅ Figure saved to: figures/processing_steps_normal.png")
