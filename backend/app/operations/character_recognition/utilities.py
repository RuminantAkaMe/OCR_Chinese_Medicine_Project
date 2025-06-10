import os
import json
from installRequirements import installRequirements
from textRecognition import recognize_and_save


class Utilities:
    # Returns a list of image file paths from the given directory.
    @staticmethod
    def get_image_files(input_dir, image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
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
            print(f"An unexpected error occurred while fetching image files: {e}")
            return []

    @staticmethod
    def process_all_images(input_dir, save_img_path, save_json_path,model_name):
        if not input_dir or not save_img_path or not save_json_path:
            print("Error: Input directory, save image path, or save JSON path is not set.")
            return
        try:
            image_files = Utilities.get_image_files(input_dir)
            if not image_files:
                print("No image files found. Exiting.")
                return

            all_results = []

            for img_path in image_files:
                try:
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
            except Exception as e:
                print(f"Error saving results to JSON file '{save_json_path}': {e}")

        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")

    @staticmethod
    def load_config():
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
            print(f"An unexpected error occurred while loading config: {e}")
            return {}

    @staticmethod
    def get_paths_from_config(config):
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
