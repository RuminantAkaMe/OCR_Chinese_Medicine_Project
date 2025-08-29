# in bash call: python backend/app/operations/word_recognition_src/model/kfold_train.py
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# kfold_train.py
"""
Grouped 5-Fold Cross-Validation for OCR word-recognition training.

What this does
--------------
- Discovers all page datasets under ./data/page*/**/*_train.jsonl (and train.jsonl).
- Builds a single HF Dataset and assigns a *group id* per sample based on the
  base page name (e.g., "page15" groups "page15", "page15f", ... together).
- Uses GroupKFold(n_splits=5) so that **all variants of the same page** are
  *always* in the same fold (prevents leakage from augmentations).
- For each fold:
  * writes train/val JSONL files,
  * trains with train_llava.py,
  * evaluates with eval_llava_metrics.py on the **val split**,
    storing checkpoints/logs per-fold.

Why grouped Cross-Validation (important)
--------------------------
Augmented copies of a page (e.g., "page15f") are highly similar to the original.
If they land in different folds, validation sees near-duplicates from training
→ over-optimistic metrics. Grouped CV keeps originals + augmentations together,
so validation is on *unseen pages*.
"""

import json
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.model_selection import GroupKFold

N_SPLITS = 5

BASE_DIR = Path(__file__).parent.parent         # folder with this script
DATA_DIR = BASE_DIR / "data"                    # expects ./data/...
CHECKPOINTS_ROOT = BASE_DIR / "checkpoints_cv"  # per-fold outputs live here

# ---------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------
def find_jsonls():
    files = []
    # typical patterns
    files += list(DATA_DIR.glob("page*/**/*_train.jsonl"))
    files += list(DATA_DIR.glob("page*/**/train.jsonl"))
    # de-dup & sort for stability
    files = sorted(set(f for f in files if f.is_file()))
    if not files:
        raise FileNotFoundError("No training JSONL files found under data/page*/")
    return files

def base_group_name(page_dir_name: str) -> str:
    """
    Collapse 'page15f' → 'page15' to tie augmented variants into the same group.
    If no match, just return the folder name.
    """
    m = re.match(r"(page\d+)", page_dir_name)
    return m.group(1) if m else page_dir_name

def load_all_data_and_groups(files):
    records = []
    groups = []
    for jf in files:
        page_dir = jf.parent.name  # e.g., 'page15f'
        group = base_group_name(page_dir)
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                groups.append(group)
    ds = Dataset.from_list(records)
    return ds, np.array(groups)

def save_subset_jsonl(dataset: Dataset, indices, out_path: Path):
    subset = dataset.select(list(indices))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in subset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# Training & evaluation orchestration
# ---------------------------------------------------------------------
def run_fold(fold: int, train_file: Path, val_file: Path):
    """
    - Copies train_file to BASE_DIR/data/train.jsonl for train_llava.py
    - Runs training with Current Working Directory at fold_dir (so checkpoints/logs go per-fold)
    - Copies val_file to fold_dir/data/train.jsonl, then runs eval there
    - Restores any previous BASE_DIR/data/train.jsonl backup
    """
    fold_dir = CHECKPOINTS_ROOT / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare training data where train_llava.py expects it:
    #    train_llava.py reads Path(__file__).parent.parent / "data" / "train.jsonl"
    #    → i.e. BASE_DIR/data/train.jsonl (relative to the script location, not Current Working Directory)
    target_train = DATA_DIR / "train.jsonl"
    backup = None
    if target_train.exists():
        backup = target_train.with_suffix(".jsonl.bak")
        if backup.exists():
            backup.unlink()
        shutil.move(target_train, backup)
    shutil.copyfile(train_file, target_train)

    # 2) Train (run in fold_dir so "./checkpoints" & "./logs" are per-fold)
    subprocess.run([sys.executable, str(Path(__file__).parent / "train_llava.py")], cwd=fold_dir, check=True)

    # 3) Eval: eval_llava_metrics.py looks at "./data/train.jsonl" relative to Current Working Directory.
    #    So we give it a fold-local copy of the *validation* set:
    fold_data_dir = fold_dir / "data"
    fold_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(val_file, fold_data_dir / "train.jsonl")

    subprocess.run([sys.executable, str(Path(__file__).parent / "eval_llava_metrics.py")], cwd=fold_dir, check=True)

    # 4) Restore
    target_train.unlink(missing_ok=True)
    if backup and backup.exists():
        shutil.move(backup, target_train)

def main():
    files = find_jsonls()
    dataset, groups = load_all_data_and_groups(files)

    splitter = GroupKFold(n_splits=N_SPLITS)
    print(f"Discovered {len(files)} files; total samples: {len(dataset)}; unique groups: {len(set(groups))}")

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.arange(len(dataset)), groups=groups)):
        # Materialize per-fold datasets so we have a reproducible record
        fold_train = DATA_DIR / f"train_fold{fold}.jsonl"
        fold_val = DATA_DIR / f"val_fold{fold}.jsonl"
        save_subset_jsonl(dataset, tr_idx, fold_train)
        save_subset_jsonl(dataset, va_idx, fold_val)

        print(f"\n=== Fold {fold} ===")
        print(f"Train: {len(tr_idx)} samples | Val: {len(va_idx)} samples | Groups in val: {sorted(set(groups[va_idx]))}")

        run_fold(fold, fold_train, fold_val)

if __name__ == "__main__":
    main()
