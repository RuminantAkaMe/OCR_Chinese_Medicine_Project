import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_path
import json
from pathlib import Path


def convert_pdf_to_images(pdf_path, output_folder):
    # First, let's clean up any existing images from previous runs
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder)

    # Convert each page of the PDF to a separate image
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []

    # Save each page as a JPEG file
    for i, page in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i}.jpg')
        page.save(image_path, 'JPEG')
        image_paths.append(image_path)

    return image_paths


def remove_red_circles(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Dilate mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Inpaint
    img_no_red = cv2.inpaint(img, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_no_red


def group_boxes_into_vertical_lines(boxes, x_threshold=50):
    boxes = sorted(boxes, key=lambda b: b[0])
    lines = []
    current_line = []

    for box in boxes:
        if not current_line:
            current_line.append(box)
        else:
            if abs(box[0] - current_line[-1][0]) < x_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

    if current_line:
        lines.append(current_line)

    return lines


def predict(model_path, images_folder, results_folder, coords_folder):
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(coords_folder, exist_ok=True)
    
    # Load YOLO model
    model = YOLO(model_path)

    # Get all JPG images
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])

    # Colors for vertical lines
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path)

        # Remove red circles
        img = remove_red_circles(img)
        
        # Save cleaned image temporarily
        temp_clean_path = os.path.join(images_folder, f"clean_{img_file}")
        cv2.imwrite(temp_clean_path, img)

        # Run YOLO detection
        results = model(temp_clean_path, conf=0.3, verbose=False)
        
        # Remove temporary file
        os.remove(temp_clean_path)
        
        # Extract bounding boxes
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes_xyxy) == 0:
            print(f"[INFO] No characters detected in {img_file}")
            # Still save the image even if no detections
            save_path = os.path.join(results_folder, img_file)
            cv2.imwrite(save_path, img)
            continue

        # Group into vertical lines
        lines = group_boxes_into_vertical_lines(boxes_xyxy, x_threshold=50)

        # Draw boxes
        for idx, line in enumerate(lines):
            color = colors[idx % len(colors)]
            for box in line:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Save result
        save_path = os.path.join(results_folder, img_file)
        cv2.imwrite(save_path, img)
        
        # Save coordinates
        coords_path = os.path.join(coords_folder, img_file.replace('.jpg', '.json'))
        with open(coords_path, 'w') as f:
            json.dump(boxes_xyxy.tolist(), f)

        print(f"[INFO] Saved detection result for {img_file}")


def run(pdf_path, output_folder):
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    
    # Set up folders relative to script location
    temp_images_folder = script_dir / 'data' / 'temp_images'
    results_folder = Path(output_folder)
    trained_model_path = script_dir / 'best.pt'  # Model in same folder as script
    coords_folder = results_folder / 'coords'

    # Create temp folder
    temp_images_folder.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting PDF to images...")
    # Step 1: Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, str(temp_images_folder))
    
    print(f"[INFO] Running character detection...")
    # Step 2: Run detection
    predict(str(trained_model_path), str(temp_images_folder), str(results_folder), str(coords_folder))

    print(f"[INFO] Process completed. Results saved to {results_folder}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2]
        run(pdf_path, output_path)
    else:
        run('Datasets/1800.pdf', 'results_7')
