import os
import json
from glob import glob
from pathlib import Path
import sys


def merge_json_files(input_folder, output_file):
    merged_data = []

    # Find all JSON files in the folder
    json_files = glob(os.path.join(input_folder, '*.json'))

    if not json_files:
        print("No JSON files found in the folder.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Ensure each entry is a dict before adding
                if isinstance(data, dict):
                    merged_data.append(data)
                elif isinstance(data, list):
                    merged_data.extend(data)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Write merged output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        print(f"Merged {len(json_files)} files into {output_file}")

        # Delete original JSON files after successful merge
        for file_path in json_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: failed to delete {file_path}: {e}")
        print("Deleted original JSON files.")

    except Exception as e:
        print(f"Error writing merged file: {e}")



