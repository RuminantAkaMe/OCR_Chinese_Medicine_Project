from installRequirements import installRequirements
from utilities import Utilities
from OCR_Chinese_Medicine_Project.backend.app.operations.character_recognition.dataset_preparation import jsons_to_txt
import sys


def setup_environment():
    """Install required packages."""
    try:
        installRequirements()
    except Exception as e:
        print(f"Error during environment setup: {e}")


def process_images(config):
    """Process images based on the configuration."""
    try:
        input_dir, save_img_path, save_json_path, model_name, _ = Utilities.get_paths_from_config(config)
        Utilities.process_all_images(input_dir, save_img_path, save_json_path, model_name)
    except Exception as e:
        print(f"Error during image processing: {e}")


def convert_json_to_txt():
    """Convert JSON files to a text file."""
    try:
        src = sys.argv[1] if len(sys.argv) > 1 else "./"
        dst = sys.argv[2] if len(sys.argv) > 2 else None
        keys_to_keep = ["input_path", "rec_text", "rec_score"]
        jsons_to_txt(src, dst, keep_keys=keys_to_keep)
    except Exception as e:
        print(f"Error during JSON to text conversion: {e}")


def main():
    """Main entry point of the program."""
    try:
        config = Utilities.load_config()
        process_images(config)
        convert_json_to_txt()
    except Exception as e:
        print(f"An error occurred in the main program: {e}")


if __name__ == "__main__":
    main()