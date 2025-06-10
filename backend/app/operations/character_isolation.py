import json
import os
from PIL import Image


COORDS_DIR = 'coords'
RAW_DIR    = 'raw_chars'
IMAGES_DIR = 'temp_images'
OUT_DIR    = 'isolated_chars'
TARGET_SIZE = (64, 64)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


for fn in sorted(os.listdir(COORDS_DIR)):
    if not fn.endswith('.json'):
        continue

    page_idx = fn.split('_')[1].split('.')[0]  
    coords_path = os.path.join(COORDS_DIR, fn)
    img_path    = os.path.join(IMAGES_DIR, f'page_{page_idx}.jpg')
    
    img = Image.open(img_path).convert('L')    

    boxes = json.load(open(coords_path, 'r'))

    for i, (x1, y1, x2, y2) in enumerate(boxes):
         char_im = img.crop((x1, y1, x2, y2))
         
         raw_fn = f'page_{page_idx}_char_{i:03d}.png'
         char_im.save(os.path.join(RAW_DIR, raw_fn))

         
         resized_im = char_im.resize(TARGET_SIZE, Image.BILINEAR)
         out_fn = f'page_{page_idx}_char_{i:03d}.png'
         resized_im.save(os.path.join(OUT_DIR, out_fn))
   
    print(f"[INFO] Done: page_{page_idx} with {len(boxes)} chars")
        
print("[DONE] Character cropping completed. Standard and raw images have been generated.")