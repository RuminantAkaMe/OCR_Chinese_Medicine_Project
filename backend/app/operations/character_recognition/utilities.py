import os
import json
import shutil
from pathlib import Path
from installRequirements import installRequirements
from textRecognition import recognize_and_save


class Utilities:
    @staticmethod
    def get_image_files(input_dir, image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        """
        Returns a list of image file paths from the given directory.

        Args:
            input_dir (str): Directory where image files are stored.
            image_extensions (tuple): Tuple of valid image file extensions.

        Returns:
            list: List of image file paths.
        """
        try:
            return [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith(image_extensions)
            ]
        except FileNotFoundError:
            print(f"Error: The directory '{input_dir}' does not exist.")
            return []
        except Exception as e:
            print(f"Unexpected error while fetching image files: {e}")
            return []

    @staticmethod
    def process_all_images(input_dir, save_img_path, model_name):
        """
        Processes all images from the input directory with the specified OCR model.
        Saves the per-image results into `save_img_path`. No combined summary JSON
        is produced by this workflow.

        Args:
            input_dir (str): Directory with the input images.
            save_img_path (str): Directory to save processed images/results.
            model_name (str): Identifier for the OCR model.
        """
        if not input_dir or not save_img_path:
            print("Error: Input directory or save image path is not set.")
            return

        image_files = Utilities.get_image_files(input_dir)
        if not image_files:
            print("No image files found. Exiting.")
            return

        # Ensure output directory exists and clear previous results there
        save_dir = Path(save_img_path)
        if save_dir.exists():
            removed = 0
            for child in save_dir.iterdir():
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                        removed += 1
                    elif child.is_dir():
                        shutil.rmtree(child)
                        removed += 1
                except Exception as e:
                    print(f"Warning: failed to remove '{child}': {e}")
            print(f"Cleared {removed} previous result files/dirs from: {save_dir}")
        else:
            save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            try:
                # The recognize_and_save function handles saving individual results
                # (per-image JSON files and any annotated images). We do not
                # aggregate results into a single summary JSON in this workflow.
                recognize_and_save(
                    image_path=img_path,
                    save_img_path=save_img_path,
                    model_name=model_name
                )
            except Exception as e:
                print(f"Error processing image '{img_path}': {e}")

        # Processing complete — individual results are saved into `save_img_path`.
        print(f"Processing complete. Individual results saved to: {save_img_path}")

    