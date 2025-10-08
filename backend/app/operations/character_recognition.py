from pathlib import Path
import subprocess
import os
from typing import Optional


def run(api_input_path: Optional[str] = None) -> str:
    """
    This function is called by API calls.
    You can specifiy the path to your script and environment here to run it.
    """
    # the parent folder: /operations
    operations_dir = Path(__file__).resolve().parent
    app_dir = operations_dir.parent
    backend_dir = app_dir.parent

    # the path to output folder of the previous stage --> YOUR INPUT
    # If the API provides a path, prefer it (it may be a single file or folder).
    if api_input_path:
        input_path = Path(api_input_path)
    else:
        input_path = operations_dir / "character_segmentation_src" / "data" / "output"
    # REQUIRED: put here the path to the file you want to present in the UI 
    presentation_path = operations_dir / "character_recognition_src" / "data" / "output" 
    # REQUIRED: create the output directory if not already existing and put your RESULTS here --> INPUT OF NEXT STAGE
    output_path = operations_dir / "character_recognition_src" / "data" / "output"

    # path to your environment
    env_python = backend_dir / ".venv310" / "Scripts" / "python.exe" 
    # path to your script
    script = os.path.abspath( operations_dir / "character_recognition_src" / "character_recognition_main.py")

    # start your script in your environment and pass input/output as CLI args
    # so the script can read them via sys.argv[1], sys.argv[2]
    subprocess.run([str(env_python), str(script), str(input_path), str(output_path)], check=True)
    

    return str(presentation_path)
