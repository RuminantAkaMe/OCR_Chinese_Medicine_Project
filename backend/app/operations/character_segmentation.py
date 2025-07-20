from pathlib import Path
import subprocess
import os

def run() -> str:
    """
    This function is called by API calls.
    You can specifiy the path to your script and environment here to run it.
    """
    # the parent folder: /operations
    operations_dir = Path(__file__).resolve().parent
    app_dir = operations_dir.parent
    backend_dir = app_dir.parent

    # the path to output folder of the previous stage --> YOUR INPUT
    input_path = operations_dir / "character_detection_src" / "data" / "output"
    # REQUIRED: put here the path to the file you want to present in the UI 
    presentation_path = operations_dir / "character_segmentation_src" / "data" / "output" / "000.png"
    # REQUIRED: create the output directory if not already existing and put your RESULTS here --> INPUT OF NEXT STAGE
    output_path = operations_dir / "character_segmentation_src" / "data" / "output"

    # path to your environment
    env_python = backend_dir / ".venv310" / "Scripts" / "python.exe" 
    # path to your script
    script = os.path.abspath( operations_dir / "character_segmentation_src" / "empty_placeholder_script.py")
    # start your script in your environment
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # you can ad arguments if you want:  subprocess.run([env_python, script, input_path, output_path], check=True)
    subprocess.run([env_python, script], check=True) 
    

    return str(presentation_path)
