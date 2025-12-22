import cv2
import numpy as np

def crop_to_content(image_path, padding=20):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Threshold/Invert
    # Since the arrows are dark on a light background, we use THRESH_BINARY_INV
    # This makes the arrows white (255) and everything else black (0)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 4. Find all coordinates of the arrow pixels
    coords = cv2.findNonZero(thresh)

    # 5. Get the bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(coords)
    if coords is not None:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 6. Crop the original image with a little bit of breathing room (padding)
    # We use max/min to ensure we don't go outside the image boundaries
    start_y = max(0, y - padding)
    end_y = min(img.shape[0], y + h + padding)
    start_x = max(0, x - padding)
    end_x = min(img.shape[1], x + w + padding)

    cropped_img = img[start_y:end_y, start_x:end_x]

    # Save the result
    cv2.imwrite('cropped_arrows.png', cropped_img)
    print("Success! Cropped image saved.")

# Run it
crop_to_content('./sample_ui/level2.jpg')