from installRequirements import installRequirements
from utilities import Utilities
from excelfile import jsons_to_txt
import sys

if __name__ == "__main__":
    try:
        # installRequirements()
        config = Utilities.load_config()
        input_dir, save_img_path, save_json_path, model_name, excel_path = Utilities.get_paths_from_config(config)
        Utilities.process_all_images(input_dir, save_img_path, save_json_path, model_name)
        
        src = sys.argv[1] if len(sys.argv) > 1 else "./"
        dst = sys.argv[2] if len(sys.argv) > 2 else None
   
        keys_to_keep = ["input_path", "rec_text", "rec_score"]
        jsons_to_txt(src, dst, keep_keys=keys_to_keep)

    except Exception as e:
        print(f"An error occurred in the main program: {e}")