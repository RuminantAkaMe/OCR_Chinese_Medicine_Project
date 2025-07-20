from pathlib import Path
import subprocess
import os

def run() -> str:
    """
    Runs the sliding window inference script for word recognition,
    and returns the path to output.json for presentation.

    Returns:
        str: Absolute path to word recognition output.json
    """
    # operations_dir = /.../backend/app/operations
    operations_dir = Path(__file__).resolve().parent
    app_dir = operations_dir.parent
    backend_dir = app_dir.parent

    # Path to output of previous stage (not used in this script directly)
    input_path = operations_dir / "character_recognition_src" / "data" / "output"

    # Final output used by UI
    presentation_path = operations_dir / "word_recognition_src" / "data" / "output" / "output_full.json"

    # Ensure output dir exists
    output_path = operations_dir / "word_recognition_src" / "data" / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Path to Python environment
    env_python = backend_dir / ".venv310" / "Scripts" / "python.exe" 

    # Path to the sliding window inference script
    script = operations_dir / "word_recognition_src" / "model" / "sliding_window_inference.py"

    # Define parameters
    window_size = "6"
    step_size = "3"  # or leave out to auto-use window_size // 2 in the script

    # Run the script
    subprocess.run(
        [env_python, str(script), "--window-size", window_size, "--step", step_size],
        check=True
    )

    return str(presentation_path)










