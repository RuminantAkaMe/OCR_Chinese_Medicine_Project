import os
import json
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
    def process_all_images(input_dir, save_img_path, save_json_path, model_name):
        """
        Processes all images from the input directory with the specified OCR model.
        Saves the per-image results and writes a summary JSON file.

        Args:
            input_dir (str): Directory with the input images.
            save_img_path (str): Directory to save processed images/results.
            save_json_path (str): Path to the summary JSON file.
            model_name (str): Identifier for the OCR model.
        """
        if not input_dir or not save_img_path or not save_json_path:
            print("Error: Input directory, save image path, or save JSON path is not set.")
            return

        image_files = Utilities.get_image_files(input_dir)
        if not image_files:
            print("No image files found. Exiting.")
            return

        all_results = []
        for img_path in image_files:
            try:
                # The recognize_and_save function handles saving individual results.
                result = recognize_and_save(
                    image_path=img_path,
                    save_img_path=save_img_path,
                    model_name=model_name
                )
                all_results.append({
                    "image": os.path.basename(img_path),
                    "result": result
                })
            except Exception as e:
                print(f"Error processing image '{img_path}': {e}")

        try:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Summary JSON saved to: {save_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON file '{save_json_path}': {e}")

    @staticmethod
    def load_config():
        """
        Load configuration from the 'config.json' file.

        Returns:
            dict: Configuration dictionary or empty dict on error.
        """
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print("Error: 'config.json' file not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: 'config.json' file is not a valid JSON.")
            return {}
        except Exception as e:
            print(f"Unexpected error while loading config: {e}")
            return {}

    @staticmethod
    def get_paths_from_config(config):
        """
        Extract required paths and model name from the configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            tuple: (input_dir, save_img_path, save_json_path, model_name, excel_path)
        """
        try:
            input_dir = config.get("input_dir", "")
            save_img_path = config.get("save_img_path", "")
            save_json_path = config.get("save_json_path", "")
            model_name = config.get("model_name", "")
            excel_path = config.get("excel_path", "")
            return input_dir, save_img_path, save_json_path, model_name, excel_path
        except Exception as e:
            print(f"Error extracting paths from config: {e}")
            return "", "", "", "", ""
