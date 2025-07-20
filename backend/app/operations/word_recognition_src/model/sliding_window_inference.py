# for inference call in bash: $ python backend/app/operations/word_recognition_src/model/sliding_window_inference.py --window-size 6 --step 2
# DONT FORGET to activate your python environment: source backend/.venv310/Scripts/activate
# sliding_window_inference.py

import json
import argparse
from pathlib import Path
from inference_llava import main as inference_main

# Paths to relevant files
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "data"
SEQUENCE_PATH = DATA_DIR / "sequence.json"
TEMP_PATH = DATA_DIR / "sequence_temp.json"
OUTPUT_PATH = DATA_DIR / "output" / "output.json"
OUTPUT_FULL_PATH = DATA_DIR / "output" / "output_full.json"

def sliding_window(sequence, window_size, step=1):
    """
    Generator that yields overlapping subsequences (windows) from the input list.
    
    Args:
        sequence (list): Full input sequence.
        window_size (int): Number of elements per window.
        step (int): Number of elements to move the window forward per iteration.

    Yields:
        tuple: (subsequence, start_index)
    """
    for i in range(0, len(sequence) - window_size + 1, step):
        yield sequence[i:i + window_size], i

def run_sliding_inference(window_size: int, step: int = 1):
    """
    Performs sliding-window inference over sequence.json using a specified window size and step.
    Each window is written to sequence_temp.json, then inference is called.
    The output from each inference is collected and appended to output_full.json.

    Args:
        window_size (int): Number of characters per window.
        step (int): Step size for sliding the window.
    """
    if not SEQUENCE_PATH.exists():
        raise FileNotFoundError(f"sequence.json not found at {SEQUENCE_PATH}")

    with open(SEQUENCE_PATH, "r", encoding="utf-8") as f:
        full_sequence = json.load(f)

    print(f"[INFO] Loaded {len(full_sequence)} sequence elements")

    output_full = []

    for window_seq, index in sliding_window(full_sequence, window_size, step):
        print(f"\n[INFO] Processing window starting at index {index}")
        print(f"[INFO] Window contains IDs: {[e['id'] for e in window_seq]}")

        # Write current window to temp file
        try:
            with open(TEMP_PATH, "w", encoding="utf-8") as f:
                json.dump(window_seq, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write sequence_temp.json: {e}")
            continue

        # Run inference on the current window
        try:
            inference_main()
        except Exception as e:
            print(f"[ERROR] Inference failed at index {index}: {e}")
            continue

        # Read output and add to full result
        if OUTPUT_PATH.exists():
            try:
                with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                    window_output = json.load(f)
                output_full.append({
                    "start_index": index,
                    "output": window_output
                })
            except Exception as e:
                print(f"[WARNING] Failed to load output for index {index}: {e}")
        else:
            print(f"[WARNING] output.json not found for index {index}")

    # Write combined results
    try:
        with open(OUTPUT_FULL_PATH, "w", encoding="utf-8") as f:
            json.dump(output_full, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Finished. Combined results written to: {OUTPUT_FULL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to write output_full.json: {e}")

def main():
    """
    Command-line entry point. Parses arguments and runs the sliding inference.
    """
    parser = argparse.ArgumentParser(description="Sliding window inference over sequence.json.")
    parser.add_argument("--window-size", type=int, required=True, help="Number of characters per window.")
    parser.add_argument("--step", type=int, default=None,
                        help="Step size for sliding window. Defaults to window_size // 2.")

    args = parser.parse_args()
    step = args.step if args.step is not None else args.window_size // 2

    run_sliding_inference(args.window_size, step)

if __name__ == "__main__":
    main()
