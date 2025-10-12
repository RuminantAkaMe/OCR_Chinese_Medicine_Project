import os
import cv2
import time
import pandas as pd
from unified_preprocess import process_crop  # make sure this module is in the same folder




BASE = os.path.dirname(os.path.abspath(__file__))

# Four evaluation subsets
EVAL_FOLDERS = {
    "double":      os.path.join(BASE, "eval_double"),
    "disjoint":    os.path.join(BASE, "eval_disjoint"),
    "noise_blank": os.path.join(BASE, "eval_noise_blank"),
    "normal":      os.path.join(BASE, "eval_normal"),
}

# Output CSV files
RESULT_CSV = os.path.join(BASE, "results_eval.csv")
SUMMARY_CSV = os.path.join(BASE, "results_eval_summary.csv")


# Function
def collect_images():
    #Scan all evaluation folders and collect image paths with their labels
    pairs = []
    for label, folder in EVAL_FOLDERS.items():
        if not os.path.exists(folder):
            print(f" Folder not found: {folder}")
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                pairs.append((os.path.join(folder, fn), label))
    print(f"Total {len(pairs)} samples collected.")
    return pairs


def evaluate():
    """Run the full evaluation pipeline."""
    samples = collect_images()
    if not samples:
        print(" No evaluation samples found. Please check eval_* folders.")
        return

    rows = []
    t0 = time.time()

    # Process each image
    for i, (path, label) in enumerate(samples):
        img = cv2.imread(path)
        if img is None:
            print(f" Failed to read image: {path}")
            continue

        start = time.time()
        res = process_crop(img, out_size=64, do_split=True)
        elapsed = time.time() - start

        # Extract quality metrics from the result dictionary
        q = res.get("quality", {})
        n_parts = len(res.get("parts", [])) if res.get("parts") else 0
        keep = res.get("keep", False)
        reason = res.get("reason", "")

        # Store all measurements for this sample
        rows.append({
            "path": os.path.basename(path),
            "label": label,
            "keep": keep,
            "reason": reason,
            "n_parts": n_parts,
            "lap_var": q.get("lap_var", 0),
            "edge_density": q.get("edge_density", 0),
            "contrast": q.get("contrast", 0),
            "occupancy": q.get("occupancy", 0),
            "time_s": elapsed,
        })

        # Progress log
        if (i + 1) % 100 == 0:
            print(f" Processed {i + 1}/{len(samples)} samples")

    # Save per-sample results
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved detailed results to {RESULT_CSV}")

    # Compute mean values grouped by label
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("label")[["lap_var", "edge_density", "contrast", "occupancy", "time_s"]]
        .mean()
        .reset_index()
        .round(4)
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f" Saved summarized results to {SUMMARY_CSV}")

    print(f"Total time: {time.time() - t0:.1f}s for {len(rows)} samples")

    # ============================================================
    # IV. Filtering Performance Evaluation (TP, FP, TN, FN) For label == "noise_blank": should be filtered (keep=False)
    # For all other labels: should be kept (keep=True)

    tp = ((df["label"] == "noise_blank") & (df["keep"] == False)).sum()
    fp = ((df["label"] != "noise_blank") & (df["keep"] == False)).sum()
    tn = ((df["label"] != "noise_blank") & (df["keep"] == True)).sum()
    fn = ((df["label"] == "noise_blank") & (df["keep"] == True)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n🔍 Filtering Performance Summary")
    print("--------------------------------")
    print(f"TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}")
    print(f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")


if __name__ == "__main__":
    evaluate()

