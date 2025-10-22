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
    script = os.path.abspath( operations_dir / "word_recognition_src" / "empty_placeholder_script.py")
    subprocess.run([env_python, script], check=True) 



    return str(presentation_path)










