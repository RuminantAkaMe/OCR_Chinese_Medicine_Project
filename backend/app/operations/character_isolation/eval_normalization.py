import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

ISO_DIR = "isolated_chars"
RAW_DIR = "raw_chars"

def list_pairs():
    files = sorted(fn for fn in os.listdir(ISO_DIR) if fn.endswith(".png"))
    for fn in files:
        iso_path = os.path.join(ISO_DIR, fn)
        raw_path = os.path.join(RAW_DIR, fn)
        if os.path.exists(raw_path):
            yield fn, iso_path, raw_path


"""
    Compute the standard deviation of character image sizes.
    A small std means all normalized characters have consistent dimensions.
    """
def compute_size_std():
    sizes = []
    for _, iso, _ in list_pairs():
        img = Image.open(iso)
        sizes.append(img.size)  # (W, H)
    ws, hs = zip(*sizes)
    return np.std(ws), np.std(hs)


"""
    Compute the average and std of character centroid offsets
    from the image center.
    This measures how well characters are spatially centered.
    """
def compute_center_offset():
    dists = []
    for _, iso, _ in list_pairs():
        img = Image.open(iso)
        w,h = img.size
        arr = np.array(img.convert("L"))
       
        mask = arr < 250  
        ys, xs = np.where(mask)
        if len(xs)==0:
            continue
        
         # Compute centroid of the character pixels
        cx, cy = xs.mean(), ys.mean()
       
       # Compute image center
        mx, my = w / 2, h / 2
        # Euclidean distance between centers
        dists.append(np.hypot(cx - mx, cy - my))
    return float(np.mean(dists)), float(np.std(dists))



"""
    Compute mean, min, and max Structural Similarity Index (SSIM)
    between raw and normalized character images.
    Measures how much structure is preserved during normalization.
    """
def compute_ssim():
    scores = []
    for _, iso, raw in list_pairs():
        a = np.array(Image.open(iso).convert("L"), dtype=np.float32)
        b = np.array(Image.open(raw).convert("L"), dtype=np.float32)
        
        if b.shape != a.shape:
            b = np.array(Image.open(raw).convert("L").resize(a.shape[::-1]), dtype=np.float32)
        score = ssim(a, b, data_range=255)
        scores.append(score)
    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))

if __name__ == "__main__":
    w_std, h_std = compute_size_std()
    print(f"Size std — width: {w_std:.2f}px, height: {h_std:.2f}px")

    mean_off, std_off = compute_center_offset()
    print(f"Center offset — mean: {mean_off:.2f}px, std: {std_off:.2f}px")

    m_ssim, min_ssim, max_ssim = compute_ssim()
    print(f"SSIM — mean: {m_ssim:.4f}, min: {min_ssim:.4f}, max: {max_ssim:.4f}")

