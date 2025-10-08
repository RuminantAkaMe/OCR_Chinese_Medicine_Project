import os
import cv2
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_path
import json


def convert_pdf_to_images(pdf_path, output_folder):
    # First, let's clean up any existing images from previous runs
    # Don't want old images messing things up
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        # Create the folder if it doesn't exist yet
        os.makedirs(output_folder)

    # Convert each page of the PDF to a separate image
    # Using 300 DPI for good quality - higher DPI means clearer text
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []

    # Save each page as a JPEG file
    for i, page in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i}.jpg')
        page.save(image_path, 'JPEG')
        image_paths.append(image_path)  # Keep track of all the image paths

    return image_paths

# Removing Red circles from the pdfs
def remove_red_circles(img):
    # Convert to HSV color space - it's easier to detect specific colors in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color appears in two ranges in HSV (wraps around at 0/180)
    # First range: 0-10 (lower reds)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    # Second range: 160-180 (upper reds)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # Combine both masks to get all red pixels
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Make the mask slightly bigger to ensure we get all the red marks
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Fill in the red areas with surrounding pixels - like magic eraser!
    img_no_red = cv2.inpaint(img, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img_no_red

# Putting boxes in each vertical line with different color
def group_boxes_into_vertical_lines(boxes, x_threshold=50):
    # Sort all boxes by their x-coordinate (left to right)
    boxes = sorted(boxes, key=lambda b: b[0])

    lines = []
    current_line = []

    # Group boxes that are vertically aligned (similar x-coordinates)
    for box in boxes:
        if not current_line:
            # Start a new line with the first box
            current_line.append(box)
        else:
            # Check if this box is close enough horizontally to be in the same line
            if abs(box[0] - current_line[-1][0]) < x_threshold:
                current_line.append(box)
            else:
                # Too far away - save current line and start a new one
                lines.append(current_line)
                current_line = [box]

    # Don't forget to add the last line!
    if current_line:
        lines.append(current_line)

    return lines

def predict(model_path, images_folder, results_folder, coords_folder):
    # Make sure our output folders exist
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(coords_folder, exist_ok=True)
    
    # Load our trained YOLO model for Chinese character detection
    model = YOLO(model_path)

    # Get all the JPG images we need to process
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    # Define some nice colors for different vertical lines
    # Each line gets its own color to make them easy to distinguish
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),      # Blue, Green, Red
        (255, 255, 0), (255, 0, 255), (0, 255, 255), # Cyan, Magenta, Yellow
        (128, 0, 0), (0, 128, 0), (0, 0, 128)       # Dark versions
    ]

    # Process each image one by one
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = cv2.imread(img_path)

        # Clean up any red circles/marks that might interfere with detection
        img = remove_red_circles(img)
        
        # Save the cleaned image temporarily
        temp_clean_path = os.path.join(images_folder, f"clean_{img_file}")
        cv2.imwrite(temp_clean_path, img)

        # Run YOLO to detect Chinese characters
        # Use file path instead of numpy array to avoid version issues
        # Using direct call method which is more compatible
        results = model(temp_clean_path, conf=0.3, verbose=False)
        
        # Remove temporary file
        os.remove(temp_clean_path)
        
        # Extract bounding boxes
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

        # Skip this image if no characters were found
        if len(boxes_xyxy) == 0:
            print(f"[INFO] No characters detected in {img_file}")
            continue

        # Group the detected characters into vertical lines (columns of text)
        lines = group_boxes_into_vertical_lines(boxes_xyxy, x_threshold=50)

        # Draw boxes around each character, coloring by which line they're in
        for idx, line in enumerate(lines):
            color = colors[idx % len(colors)]  # Cycle through colors if we have many lines
            for box in line:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Save the image with all the detection boxes drawn on it
        save_path = os.path.join(results_folder, img_file)
        cv2.imwrite(save_path, img)
        
        # Also save the coordinates as JSON for later use
        coords_path = os.path.join(coords_folder, img_file.replace('.jpg', '.json'))
        with open(coords_path, 'w') as f:
            json.dump(boxes_xyxy.tolist(), f)

        print(f"[INFO] Saved detection result for {img_file} with vertical lines in {results_folder}")

def run(pdf_path):
    # Set up all our folders
    temp_images_folder = 'temp_images'  # Where we'll store the PDF pages as images
    results_folder = 'results_7'        # Where the detection results go
    trained_model_path = 'best.pt'  # Our trained model
    coords_folder = 'coords'            # Where we save the coordinate data

    # Step 1: Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)
    
    # Step 2: Run detection on all the images
    predict(trained_model_path, temp_images_folder, results_folder, coords_folder)

    # Step 3: Get the second page result (for some specific reason)
    # Sort the results to make sure we get them in order
    result_images = sorted(
        [f for f in os.listdir(results_folder) if f.lower().endswith('.jpg')]
    )

    # We need at least 2 images to get the second one
    if len(result_images) >= 2:
        second_image_path = os.path.join(results_folder, result_images[1])

        # Load the second image
        img = cv2.imread(second_image_path)

        # Make sure the image loaded properly
        if img is None:
            print("[ERROR] Failed to load image.")
            return None

        return img  # Return the second page with detections
    else:
        print("[ERROR] Less than 2 images in results folder.")
        return None

# Run the whole pipeline on our PDF
if __name__ == "__main__":
    run('Datasets/1800.pdf')
