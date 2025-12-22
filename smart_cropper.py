import cv2
import numpy as np
import os

def crop_central_arrows(image_path, output_path="cropped_arrows.png", padding=20):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    h, w = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours detected.")
        return

    # Filter contours
    valid_points = []
    
    # We will ignore contours that are in the top 15% or bottom 35% of the image
    top_limit = h * 0.15
    bottom_limit = h * 0.75 # Keep content only above this line (y < 0.65h)
    # Actually, let's be more specific: if the contour's center is in the "bad" zone, drop it.
    
    for cnt in contours:
        # Get bounding box of the contour
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        center_y = cy + ch / 2
        
        # Check if the contour is within the central vertical region
        if center_y > top_limit and center_y < bottom_limit:
            # Add all points of this contour to our valid list
            # reshape to (N, 2) for easier concatenation if needed, 
            # but usually we just want the bounding rect of the combined valid contours.
            valid_points.append(cnt)

    if not valid_points:
        print("No valid arrows detected in the central region after filtering.")
        return

    # Combine all valid contours to find the global bounding box
    all_valid_points = np.vstack(valid_points)
    x, y, w_box, h_box = cv2.boundingRect(all_valid_points)

    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + w_box + padding)
    y2 = min(h, y + h_box + padding)

    # Crop
    cropped_img = gray[y1:y2, x1:x2]

    cv2.imwrite(output_path, cropped_img)
    print(f"Saved cropped image to {output_path}")
    return cropped_img

if __name__ == "__main__":
    crop_central_arrows('sample_ui/level3.jpg', 'cropped_arrows.png')
