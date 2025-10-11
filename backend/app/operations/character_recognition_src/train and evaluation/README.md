Character Recognition — Training & Evaluation

This README describes how to run training and evaluation for the Character Recognition module. The training entrypoint is `train.py` (a small, PaddleOCR-based runner adapted for this project).

Location
- `backend/app/operations/character_recognition_src/train and evaluation/train.py`


Prerequisites
- Python 3.8+ (the project uses PaddlePaddle and related dependencies). Use a virtual environment.
- PaddlePaddle installed with the appropriate CUDA/CUPY support for your GPU, or the CPU-only build if you don't have a GPU.
- Required Python packages (see `backend/app/operations/character_recognition_src/requirements.txt` if present, and the top-level `backend/requirements.txt`).

Quick setup (Windows PowerShell)
1. Create and activate a venv:
```powershell
cd 'e:\ocr_code\OCR_Chinese_Medicine_Project\backend'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (Optional) Install project-specific requirements if present
if (Test-Path '.\app\operations\character_recognition_src\requirements.txt') { pip install -r .\app\operations\character_recognition_src\requirements.txt }
```

2. Ensure Paddle is installed (example for GPU):
```powershell
pip install paddlepaddle-gpu -f https://paddlepaddle.org.cn/whl/windows.html
```
Use the CPU package if you don't have a supported GPU.

How train.py works (short)
- Reads a YAML configuration via `program.preprocess(is_train=True)`.
- Builds Train / Eval dataloaders using `ppocr.data.build_dataloader`.
- Builds the model via `ppocr.modeling.architectures.build_model` and wraps for distributed training if configured.
- Builds optimizer, lr-scheduler and optional AMP (automatic mixed precision).
- Optionally uses an EarlyStopping callback (this script injects an `EarlyStopping` instance into `config['Global']['early_stopping_callback']`).
- Calls `program.train(...)` which contains the core training loop and evaluation checkpoints.

Key configuration options (found in your YAML config)
- Global:
  - `distributed` (bool): whether to use distributed training
  - `use_early_stopping` (bool): enables early stopping
  - `early_stopping_patience` (int): epochs without improvement before stopping
  - `early_stopping_min_delta` (float): minimum improvement to reset counter
  - `early_stopping_mode` ("max" or "min"): "max" for accuracy, "min" for loss
  - `restore_best_weights` (bool): if true, the best model weights are kept in memory and can be restored after training stops early
  - `save_best_model_only` (bool): if true, the script saves only the current best model to `save_model_dir`
  - `save_model_dir` (str): directory to save model checkpoints
  - `epoch_num` (int): total number of epochs
  - `use_amp`, `amp_level`, `amp_dtype`, `scale_loss`: AMP settings
- Architecture, Loss, Optimizer, PostProcess, Metric: standard PaddleOCR keys used to build the model, loss, optimizer, and metric objects.

Examples — Run training (PowerShell)
1. Basic (non-distributed) training using the project's standard configuration loader. From project backend root:
```powershell
cd 'e:\ocr_code\OCR_Chinese_Medicine_Project\backend'
# This project expects to call into the package's tooling. The training wrapper reads config via program.preprocess.
python -u .\app\operations\character_recognition_src\train and evaluation\train.py
```
2. If you have a config file and program entry expects the path via env var or CLI, set it accordingly (example pattern used in the codebase — adjust to your project's program.preprocess contract):
```powershell
$env:CONFIG_PATH = 'e:\path\to\your_config.yml'
python -u .\app\operations\character_recognition_src\train and evaluation\train.py
```

Early stopping behavior
- If enabled (`use_early_stopping`), the script constructs an `EarlyStopping` instance and places it at `config['Global']['early_stopping_callback']` for the training loop to call.
- The EarlyStopping instance tracks the best score, the epoch at which it occurred, and (if `restore_best_weights` is True) keeps the best model's weights in memory.
- If `save_best_model_only` is True, the best model (pdparams) will be written to `save_model_dir` with a filename starting `best_accuracy_epoch_`.

.



