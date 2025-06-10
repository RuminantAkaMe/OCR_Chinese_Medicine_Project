import os
import cv2
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_path
import json


def convert_pdf_to_images(pdf_path, output_folder):
    # Clear existing files in output_folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder)

    # Now convert PDF
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []

    for i, page in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i}.jpg')
        page.save(image_path, 'JPEG')
        image_paths.append(image_path)

    return image_paths

#Removing Red circles from the pdfs
def remove_red_circles(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    img_no_red = cv2.inpaint(img, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_no_red

# Putting boxes in each vertical line with different color
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

def predict(model_path, images_folder, results_folder, coords_folder ):
    os.makedirs(results_folder, exist_ok=True)
    
    model = YOLO(model_path)  # YOLO Model

    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path)

        img = remove_red_circles(img)

        results = model.predict(source=img, save=False, conf=0.3)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes_xyxy) == 0:
            print(f"[INFO] No characters detected in {img_file}")
            continue

        lines = group_boxes_into_vertical_lines(boxes_xyxy, x_threshold=50)

        for idx, line in enumerate(lines):
            color = colors[idx % len(colors)]
            for box in line:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        save_path = os.path.join(results_folder, img_file)
        cv2.imwrite(save_path, img)
        coords_path = os.path.join(coords_folder, img_file.replace('.jpg', '.json'))
        with open(coords_path, 'w') as f:
            json.dump(boxes_xyxy.tolist(), f)

        print(f"[INFO] Saved detection result for {img_file} with vertical lines in {results_folder}")

def run(pdf_path):
    temp_images_folder = 'temp_images'  # Define temporary image folder
    results_folder = 'results_7'
    trained_model_path = 'yolov10n.pt'  # Updated model path
    coords_folder = 'coords'

    image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)
    predict(trained_model_path, temp_images_folder, results_folder, coords_folder)

    # Get sorted list of image filenames
    result_images = sorted(
        [f for f in os.listdir(results_folder) if f.lower().endswith('.jpg')]
    )

    if len(result_images) >= 2:
        second_image_path = os.path.join(results_folder, result_images[1])

        # Read the image
        img = cv2.imread(second_image_path)

        if img is None:
            print("[ERROR] Failed to load image.")
            return None

        return img
    else:
        print("[ERROR] Less than 2 images in results folder.")
        return None
