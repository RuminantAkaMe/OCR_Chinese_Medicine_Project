def remove_red_circles(img):
    """
    Removes red circular areas from an image using color masking and inpainting.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for two red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Optional: Dilate the mask a little to cover edges better
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    # Inpaint (smart fill the masked regions)
    img_no_red = cv2.inpaint(img, red_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return img_no_red
