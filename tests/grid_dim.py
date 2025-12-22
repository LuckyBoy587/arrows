import cv2
import numpy as np

def get_grid_info(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter very small noise
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25]
    
    if not valid_contours:
        return 0, 0, 0, 0

    centroids_x = []
    centroids_y = []

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        centroids_x.append(cx)
        centroids_y.append(cy)

    # Helper to find clusters of coordinates (peaks in frequency)
    def analyze_axis(coords, axis_length, tolerance=15):
        if not coords:
            return 0
        
        coords = sorted(coords)
        clusters = []
        
        # Simple 1D clustering
        if coords:
            current_cluster = [coords[0]]
            for c in coords[1:]:
                # If within tolerance, group it
                if c - current_cluster[-1] < tolerance:
                    current_cluster.append(c)
                else:
                    # Store the mean of the cluster
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [c]
            clusters.append(np.mean(current_cluster))
            
        # clusters now contains the coordinate "centers" of the rows/cols
        if len(clusters) < 2:
            # Only one row/col detected.
            # Cannot determine spacing from differences.
            # Return 0 or maybe estimate from object size if needed?
            return 0
            
        # Calculate spacings between adjacent clusters
        spacings = np.diff(clusters)
        
        # User requested "average distance"
        # We use median to filter out potential jumps if a row/col is missing, 
        # but "average" is requested. Let's use median as a robust average.
        avg_spacing = np.median(spacings)
        
        return avg_spacing

    img_h, img_w = img.shape[:2]
    
    # Analyze X axis -> Cell Width
    cell_width = analyze_axis(centroids_x, img_w)
    
    # Analyze Y axis -> Cell Height
    cell_height = analyze_axis(centroids_y, img_h)

    # Fallback if we couldn't determine spacing (e.g. 1xN grid)
    # If one is missing, maybe assume square? Or use the other?
    if cell_width == 0 and cell_height > 0:
        cell_width = cell_height # Assume square cells
    elif cell_height == 0 and cell_width > 0:
        cell_height = cell_width

    # Calculate Grid Dimensions
    # Rows = Image Height / Cell Height
    # Cols = Image Width / Cell Width
    
    rows = 0
    cols = 0
    
    if cell_height > 0:
        rows = int(round(img_h / cell_height))
        
    if cell_width > 0:
        cols = int(round(img_w / cell_width))

    return rows, cols, cell_height, cell_width

if __name__ == "__main__":
    try:
        image_path = "cropped_arrows.png"
        r, c, h_dim, w_dim = get_grid_info(image_path)
        print(f"Estimated Grid: {r}x{c}")
        print(f"Estimated Cell Dimensions: Width={w_dim:.2f}, Height={h_dim:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    print(f"Dimensions: {get_grid_info(image_path)}")
