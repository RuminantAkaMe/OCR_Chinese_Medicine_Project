from installRequirements import installRequirements
from utilities import Utilities
try:
    from merge_json_files import merge_json_files
except Exception:
    merge_json_files = None
import sys
import os



_DEFAULT_MODEL_NAME = "PP-OCRv5_server_rec"

def setup_environment():
    """Install required packages."""
    try:
        installRequirements()
    except Exception as e:
        print(f"Error during environment setup: {e}")


def process_images():
    """Process images using only CLI arguments; do not read config.json.

    Expected positional arguments:
      argv[1] -> input_dir
      argv[2] -> save_img_path (directory)
      argv[3] -> save_json_path (optional; file path)
      argv[4] -> model_name (optional)

    If required args are missing the function will print an error and return.
    """
    try:
        if len(sys.argv) < 3:
            print("Error: this script requires at least two arguments: input_dir and save_img_path")
            return

        input_dir = sys.argv[1]
        save_img_path = sys.argv[2]
        model_name = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_MODEL_NAME
        # Optional 4th positional arg: merge flag ("true"/"1"/"yes"/"merge")
        merge_flag = True
        if len(sys.argv) > 4:
            m = sys.argv[4].lower()
            merge_flag = m in ("false", "False", "FALSE")

        Utilities.process_all_images(input_dir, save_img_path, model_name)

        if merge_flag:
            if merge_json_files is None:
                print("Merge requested but merge_json_files.py is not available.")
            else:
                try:
                    out_path = os.path.join(save_img_path, "character_recognition_results.json")
                    merge_json_files(save_img_path, out_path)
                except Exception as e:
                    print(f"Error during merging JSON files: {e}")
    except Exception as e:
        print(f"Error during image processing: {e}")


def main():
    """Main entry point of the program."""
    try:
        setup_environment()
        # Do not load configuration from file; read paths from CLI only
        process_images()
    except Exception as e:
        print(f"An error occurred in the main program: {e}")


if __name__ == "__main__":
    main()