import json
import os
import glob
import numpy as np
from PIL import Image
from unified_preprocess import process_crop   # import preprocess function

# Directory Configuration 
#  output folder
INPUT_DIR = os.path.join("..", "character_detection_src", "LabeledImage")

RAW_DIR    = 'raw_chars'
OUT_DIR    = 'isolated_chars'
TARGET_SIZE = (64, 64)

# Ensure required directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def isolate_chars(input_dir, raw_folder, out_folder, target_size):
    """
    Read detection output (image + .json pairs),
    crop each bounding box, and save processed 64x64 characters.
    """

    # Collect all image files
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) +
                         glob.glob(os.path.join(input_dir, "*.jpeg")) +
                         glob.glob(os.path.join(input_dir, "*.png")))

    out_files = []

    for img_path in image_files:
        json_path = img_path + ".json"
        if not os.path.exists(json_path):
            print(f"[WARN] No JSON found for {os.path.basename(img_path)}")
            continue

        # Open page image and load JSON
        img = Image.open(img_path).convert('L')
        with open(json_path, 'r') as f:
            boxes = json.load(f)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img.crop((x1, y1, x2, y2))

            # Save raw cropped image using original filename
            raw_fn = f"{os.path.basename(img_path)}_char_{i:03d}.png"
            raw_save_path = os.path.join(raw_folder, raw_fn)
            crop.save(raw_save_path)

            # Apply unified preprocessing (denoise + contrast + normalization)
            crop_cv = np.array(crop)
            result = process_crop(crop_cv, target_size)

            if result["keep"] and len(result["parts"]) > 0:
                final_img = Image.fromarray(result["parts"][0])
                out_path = os.path.join(out_folder, raw_fn)
                final_img.save(out_path)
                out_files.append(out_path)
            else:
                print(f"[WARN] Skipped {raw_fn}")

    print(f"[INFO] Cropping done, generated {len(out_files)} processed characters.")
    return out_files


if __name__ == '__main__':
    result = isolate_chars(INPUT_DIR, RAW_DIR, OUT_DIR, TARGET_SIZE)
    print("Standalone run ->", result)




