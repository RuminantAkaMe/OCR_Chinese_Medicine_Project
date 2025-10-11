
import os
import time
import random
import cv2
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# three chars folder
BASE = os.path.dirname(os.path.abspath(__file__))

folders = {
    '48×48': (os.path.join(BASE, 'raw_chars_48'), os.path.join(BASE, 'chars_48')),
    '64×64': (os.path.join(BASE, 'raw_chars'),    os.path.join(BASE, 'isolated_chars')),
    '80×80': (os.path.join(BASE, 'raw_chars_80'), os.path.join(BASE, 'chars_80')),
}

results = []
SAMPLE_N = 100

for label, (raw_dir, proc_dir) in folders.items():
    # 找到公共文件名
    raws = [f for f in os.listdir(raw_dir)  if f.endswith('.png')]
    procs= [f for f in os.listdir(proc_dir) if f.endswith('.png')]
    common = list(set(raws) & set(procs))
    sample = random.sample(common, min(SAMPLE_N, len(common)))
    
    # 1) 计算平均 SSIM
    scores = []
    for fn in sample:
        raw_img = cv2.imread(os.path.join(raw_dir, fn), cv2.IMREAD_GRAYSCALE)
        proc_img= cv2.imread(os.path.join(proc_dir, fn),cv2.IMREAD_GRAYSCALE)
        # 把 raw resize 回 proc 的尺寸
        raw_rs = cv2.resize(raw_img, (proc_img.shape[1], proc_img.shape[0]))
        score, _ = ssim(raw_rs, proc_img, full=True)
        scores.append(score)
    avg_ssim = sum(scores)/len(scores) if scores else None

    # 2) calculate the readout time
    t0 = time.time()
    for fn in sample:
        cv2.imread(os.path.join(proc_dir, fn), cv2.IMREAD_GRAYSCALE)
    avg_time = (time.time() - t0)/len(sample) if sample else None

    results.append({
        'Size': label,
        'Avg SSIM': avg_ssim,
        'Avg Read Time (s)': avg_time
    })

# get the form
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
