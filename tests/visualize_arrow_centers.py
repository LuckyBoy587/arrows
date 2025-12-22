import cv2
import sys
import os

# Add project root to path to allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_cropper import crop_central_arrows
from arrow_utils import get_arrow_head_center

def test_visualize(image_path, output_path):
    print(f"Processing image: {image_path}")
    
    # Use the existing smart cropper to get the relevant region
    # Note: crop_central_arrows also saves a file to root by default, we ignore that side effect here.
    img_gray = crop_central_arrows(image_path)
    
    if img_gray is None:
        print("Error: Could not crop/load image.")
        return

    # Convert to color for visualization
    img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Threshold to get contours (standardizing input for our detection)
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours.")

    for i, contour in enumerate(contours):
        center = get_arrow_head_center(contour)
        
        # Draw the contour in Green
        cv2.drawContours(img_vis, [contour], -1, (0, 255, 0), 2)
        
        if center:
            print(f"  Arrow {i}: Center found at {center}")
            # Draw a solid Red circle at the calculated center
            cv2.circle(img_vis, center, 4, (0, 0, 255), -1)
            
            # Draw a Yellow crosshair for precise visibility
            x, y = center
            cv2.line(img_vis, (x - 6, y), (x + 6, y), (0, 255, 255), 1)
            cv2.line(img_vis, (x, y - 6), (x, y + 6), (0, 255, 255), 1)
        else:
            print(f"  Arrow {i}: Center not found")

    cv2.imwrite(output_path, img_vis)
    print(f"Visualized result saved to: {output_path}")

if __name__ == "__main__":
    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    input_img = os.path.join(project_root, 'sample_ui', 'level2.jpg')
    output_img = os.path.join(base_dir, 'result_level2.png')
    
    if not os.path.exists(input_img):
        print(f"Error: Input file not found at {input_img}")
    else:
        test_visualize(input_img, output_img)
