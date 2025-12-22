import cv2
import numpy as np
from smart_cropper import crop_central_arrows
from arrow_utils import get_arrow_head_center
# Constants for our grid return
EMPTY = "EMPTY"
ARROW = "ARROW"
EDGE = "EDGE"

def remove_extras(img):
    h, w = img.shape[:2]
    side = min(h, w)

    start_x = w // 2 - side // 2
    start_y = h // 2 - side // 2

    cropped_img = img[start_y:start_y + side, start_x:start_x + side]
    return cropped_img


def analyze_puzzle_grid(image_path, grid_size=(5, 5)):
    # 1. Load and Preprocess
    img = crop_central_arrows(image_path)
    # Thresholding to get a binary image (black lines on white background)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. Find Contours (the arrows and lines)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a logical grid (e.g., 5x5 based on your image)
    grid = [[{"type": EMPTY} for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    print(f"Contours found: {len(contours)}")
    arrow_centers = []
    for i, contour in enumerate(contours):
        h, w = img.shape[:2]
        img_copy = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)
        cv2.namedWindow(f"Contour {i}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f"Contour {i}", img_copy)
        arrow_centers.append(get_arrow_head_center(contour))
    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass
    print("Arrow Centers Detected:", arrow_centers)
analyze_puzzle_grid("sample_ui\\level3.jpg")