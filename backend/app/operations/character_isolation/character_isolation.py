import json
import os
import glob
import numpy as np
from PIL import Image
from unified_preprocess import process_crop   # import preprocess function


# --- Directory Configuration ---
COORDS_DIR = 'coords'
RAW_DIR    = 'raw_chars'
IMAGES_DIR = 'temp_images'
OUT_DIR    = 'isolated_chars'
TARGET_SIZE = (64, 64)


# Ensure required directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def isolate_chars(images_folder, coords_folder, raw_folder, out_folder, target_size):

    # Collect all coordinate JSON files (sorted by page number)
    coord_paths = sorted(glob.glob(os.path.join(coords_folder, 'page_*.json')))
    out_files = []

    for coord_path in coord_paths:
        idx = os.path.splitext(os.path.basename(coord_path))[0].split('_')[1]
        
        # Find corresponding image for this page
        img_path = os.path.join(images_folder, f'page_{idx}.jpg')

        if not os.path.isfile(img_path):
            print(f"[WARN] Missing {img_path}")
            continue

        # Open page image and convert to grayscale
        img = Image.open(img_path).convert('L')
        boxes = json.load(open(coord_path))

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img.crop((x1, y1, x2, y2))

            # Save raw cropped image 
            raw_fn = f'page_{idx}_char_{i:03d}.png'
            crop.save(os.path.join(raw_folder, raw_fn))

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
    result = isolate_chars(IMAGES_DIR, COORDS_DIR, RAW_DIR, OUT_DIR, TARGET_SIZE)
    print("Standalone run ->", result)




