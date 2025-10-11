import cv2
import numpy as np


def process_crop(image, target_size=(64, 64)):

    try:
        # Convert to grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Gentle background correction 
        background = cv2.medianBlur(gray, 21)
        normalized = cv2.divide(gray, background, scale=255)

        # Contrast enhancement
        enhanced = cv2.convertScaleAbs(normalized, alpha=1.5, beta=10)

        # Soft threshold 
        _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Preserve stroke details
        mixed = cv2.addWeighted(enhanced, 0.7, mask, 0.3, 0)

        # Resize to 64×64 
        resized = cv2.resize(mixed, target_size, interpolation=cv2.INTER_AREA)

        # Normalize contrast again 
        final = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)

        return {"parts": [final], "keep": True}

    except Exception as e:
        print(f"[WARN] Preprocessing failed: {e}")
        return {"parts": [], "keep": False}
