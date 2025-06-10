import json
import os
from PIL import Image
import glob


COORDS_DIR = 'coords'
RAW_DIR    = 'raw_chars'
IMAGES_DIR = 'temp_images'
OUT_DIR    = 'isolated_chars'
TARGET_SIZE = (64, 64)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def isolate_chars(images_folder, coords_folder, raw_folder, out_folder, target_size):

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    coord_paths = sorted(glob.glob(os.path.join(coords_folder, 'page_*.json')))
    out_files = []
    for coord_path in coord_paths:
        idx = os.path.splitext(os.path.basename(coord_path))[0].split('_')[1]
        img_path = os.path.join(images_folder, f'page_{idx}.jpg')
        if not os.path.isfile(img_path):
            print(f"[WARN] Missing {img_path}")
            continue

        img = Image.open(img_path).convert('L')
        boxes = json.load(open(coord_path))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img.crop((x1, y1, x2, y2))

            # save raw image
            raw_fn = f'page_{idx}_char_{i:03d}.png'
            crop.save(os.path.join(raw_folder, raw_fn))
            # save resized image
            resized = crop.resize(target_size, Image.BILINEAR)
            out_path = os.path.join(out_folder, raw_fn)
            resized.save(out_path)
            out_files.append(out_path)

    print(f"[INFO] Cropping done, generated {len(out_files)} images")
    return out_files

def run(
    images_folder: str = IMAGES_DIR,
    coords_folder: str = COORDS_DIR,
    raw_folder: str = RAW_DIR,
    out_folder: str = OUT_DIR,
    target_size: tuple = TARGET_SIZE
) -> list[str]:
    return isolate_chars(images_folder, coords_folder, raw_folder, out_folder, target_size)

if __name__ == '__main__':
    result = isolate_chars(IMAGES_DIR, COORDS_DIR, RAW_DIR, OUT_DIR, TARGET_SIZE)
    print("Standalone run ->", result)
