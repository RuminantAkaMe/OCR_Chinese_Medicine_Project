"""
dataset_preparation.py

Helpers for extracting individual character PNGs and a label file from
GNT-format dataset files. The GNT format is a simple binary container used
by some Chinese character datasets where each sample contains a tag (the
character code), width/height and raw grayscale bitmap data.

This module intentionally keeps processing minimal and robust: corrupt
samples are skipped so a single malformed entry does not stop batch
extraction.
"""

import os
import struct
from pathlib import Path
from PIL import Image

def extract_valid_samples_from_gnt(gnt_path, output_img_dir, label_file_handle, start_index=0):
    """Extract valid Chinese-character samples from a single .gnt file.

    Parameters
    ----------
    gnt_path : str or Path
        Path to the .gnt file to read.
    output_img_dir : Path
        Directory where extracted PNGs will be written. The caller should
        ensure the directory exists.
    label_file_handle : file-like
        An open text file handle (UTF-8) to which label lines are written in
        the format: "<image_filename>\t<char>\n".
    start_index : int
        Index to use when generating sequential image filenames. This allows
        multiple .gnt files to be processed into a single contiguous set.

    Returns
    -------
    tuple
        (total_count, valid_count) where total_count is the number of samples
        encountered and valid_count is the number successfully extracted.
    """

    with open(gnt_path, 'rb') as f:
        file_size = os.path.getsize(gnt_path)
        total_count = 0
        valid_count = 0

        # Iterate over samples until EOF
        while f.tell() < file_size:
            sample_start = f.tell()
            try:
                # Read sample size (4 bytes, little-endian unsigned int)
                sample_size_bytes = f.read(4)
                if len(sample_size_bytes) < 4:
                    # Reached truncated footer; stop parsing
                    break
                sample_size = struct.unpack('<I', sample_size_bytes)[0]

                # Read tag code (2 bytes). The format stores the two bytes in
                # an order that requires swapping before GB2312 decoding.
                tag_code_raw = f.read(2)
                tag_code = bytes([tag_code_raw[1], tag_code_raw[0]])
                char = tag_code.decode('gb2312')

                # Ensure it's a common Chinese character (CJK Unified Ideographs)
                if not ('\u4e00' <= char <= '\u9fff'):
                    raise ValueError("Invalid char")

                # Read image width/height (2 bytes each, little-endian unsigned short)
                width = struct.unpack('<H', f.read(2))[0]
                height = struct.unpack('<H', f.read(2))[0]

                # Read raw bitmap data (width * height bytes expected)
                bitmap = f.read(width * height)
                if len(bitmap) != width * height:
                    raise ValueError("Incomplete bitmap")

                # Convert bytes to a PIL image. GNT uses bottom-left origin so
                # flip/rotate to produce an upright PNG for usual viewing.
                img = Image.frombytes('L', (width, height), bitmap)
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img = img.transpose(Image.Transpose.ROTATE_270)

                # Save image with a sequential filename
                img_name = f"sample_{start_index + valid_count:06d}.png"
                img_path = output_img_dir / img_name
                img.save(img_path)

                # Append a tab-separated label line
                label_file_handle.write(f"{img_name}\t{char}\n")

                valid_count += 1

            except Exception:
                # Skip corrupt or undecodable sample by seeking to the next
                # sample boundary using the declared sample_size. This allows
                # processing to continue even when a single sample is bad.
                f.seek(sample_start + sample_size, 0)

            total_count += 1

        return total_count, valid_count


def process_gnt_folder(gnt_folder, output_root):
    gnt_folder = Path(gnt_folder)
    output_root = Path(output_root)
    img_output_dir = output_root / "Gnt1.1TestImages"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    label_file_path = output_root / "Gnt1.1TestImages.txt"
    total_samples = 0
    total_valid = 0
    next_index = 0

    gnt_files = sorted([f for f in gnt_folder.glob("*.gnt")])

    with open(label_file_path, 'w', encoding='utf-8') as label_f:
        for gnt_file in gnt_files:
            print(f"📦 Processing {gnt_file.name}...")
            total, valid = extract_valid_samples_from_gnt(
                gnt_file, img_output_dir, label_f, start_index=next_index
            )
            total_samples += total
            total_valid += valid
            next_index += valid

    print("\n✅ Batch extraction completed.")
    print(f"Total files processed: {len(gnt_files)}")
    print(f"Total samples: {total_samples}")
    print(f"Valid Chinese samples saved: {total_valid}")
    print(f"Output folder: {output_root}")




if __name__ == "__main__":
    gnt_input_folder = "/home/woody/vlbi/vlbi102v/Gnt1.1Test"  
    output_folder = "/home/woody/vlbi/vlbi102v/Gnt1.1Test"

    process_gnt_folder(gnt_input_folder, output_folder)
