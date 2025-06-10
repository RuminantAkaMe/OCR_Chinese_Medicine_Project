from installRequirements import installRequirements
from utilities import Utilities

if __name__ == "__main__":
    try:
        installRequirements()
        config = Utilities.load_config()
        input_dir, save_img_path, save_json_path,model_name = Utilities.get_paths_from_config(config)
        Utilities.process_all_images(input_dir, save_img_path, save_json_path, model_name)
    except Exception as e:
        print(f"An error occurred in the main program: {e}")