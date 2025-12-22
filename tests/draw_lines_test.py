import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_cropper import crop_central_arrows

def get_skeleton(img_binary):
    """
    Computes the skeleton of a binary image using morphological operations.
    """
    skel = np.zeros(img_binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = img_binary.copy()
    
    while True:
        # Open: erode then dilate
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        # Subtract open from original to get the skeletal part
        temp = cv2.subtract(eroded, temp)
        # Add to skeleton
        skel = cv2.bitwise_or(skel, temp)
        # Erode the original
        eroded = cv2.erode(eroded, element)
        
        # Check if done
        if cv2.countNonZero(eroded) == 0:
            break
    return skel

def get_pruned_path(skeleton_mask):
    """
    Finds the longest path in the skeleton and prunes the arrow head
    by cutting at the junction point (branching point).
    """
    # Get all non-zero points: coordinates are (y, x)
    pts = np.column_stack(np.where(skeleton_mask > 0))
    if len(pts) == 0:
        return []

    # Build graph: (r, c) -> list of neighbors
    pts_set = set(map(tuple, pts))
    adj = {p: [] for p in pts_set}
    
    # 8-connectivity
    for p in pts_set:
        r, c = p
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nbr = (r + dr, c + dc)
                if nbr in pts_set:
                    adj[p].append(nbr)

    # Helper: BFS to find farthest node
    def bfs_farthest(start_node):
        q = [(start_node, 0)]
        visited = {start_node}
        farthest = start_node
        max_dist = 0
        parent = {start_node: None}
        
        idx = 0
        while idx < len(q):
            u, dist = q[idx]
            idx += 1
            if dist > max_dist:
                max_dist = dist
                farthest = u
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append((v, dist + 1))
        return farthest, max_dist, parent

    # 1. Find diameter path
    start_candidates = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    start_node = start_candidates[0] if start_candidates else next(iter(pts_set))
    
    u, _, _ = bfs_farthest(start_node)
    v, _, parent = bfs_farthest(u)
    
    # Reconstruct raw path (y, x)
    raw_path = []
    curr = v
    while curr is not None:
        raw_path.append(curr)
        curr = parent[curr]
        
    # 2. Pruning Logic: Remove head
    # Identify junctions on the path (degree > 2)
    path_junctions = [i for i, p in enumerate(raw_path) if len(adj[p]) > 2]
    
    if path_junctions:
        # We assume the junction separates the shaft from the head tip.
        # The path is likely [ShaftStart ... Junction ... Tip] (or vice versa).
        # We want to cut at the junction and keep the longer side.
        
        # Junctions might be a cluster of pixels, so we take the range.
        first_j = path_junctions[0]
        last_j = path_junctions[-1]
        
        # Segment 1: Start to first junction
        seg1 = raw_path[:first_j]
        # Segment 2: Last junction to End
        seg2 = raw_path[last_j+1:]
        
        # If segments are empty (e.g. junction is at endpoint?), handle gracefully
        # Usually keeping the "junction" pixel itself in the shaft is fine,
        # but strictly cutting it out is also okay. Let's include the junction in the retained part
        # to ensure connectivity to the "branch point".
        
        seg1_inclusive = raw_path[:last_j+1] # From start to end of junction cluster
        seg2_inclusive = raw_path[first_j:]  # From start of junction cluster to end
        
        # Actually, simpler: Compare lengths of exclusive parts to decide "main body"
        # then attach the junction to it.
        if len(seg1) > len(seg2):
            # Keep start -> junction
            # We cut OFF the junction to be safe? Or keep it?
            # Keeping it ensures we reach the "base" of the triangle.
            raw_path = raw_path[:last_j+1]
        else:
            # Keep junction -> end
            raw_path = raw_path[first_j:]

    # 3. Convert to (x, y) for OpenCV
    final_path = [(p[1], p[0]) for p in raw_path]
    return final_path

def draw_midlines_with_l_turns(contours, img_shape):
    """
    Takes a list of contours, finds the "middle" line (skeleton),
    and draws it using strictly horizontal and vertical segments.
    """
    # 1. Create a binary mask where contours are filled white
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # 2. Skeletonize the filled shapes
    skeleton = get_skeleton(mask)
    
    line_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # 3. Process each connected component of the skeleton separately
    num_labels, labels = cv2.connectedComponents(skeleton)
    
    for i in range(1, num_labels + 1):
        # Extract single component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Get the pruned path (shaft only)
        path_pts = get_pruned_path(component_mask)
        
        if len(path_pts) < 2:
            continue
            
        # Convert to numpy format for approxPolyDP: (N, 1, 2)
        cnt = np.array(path_pts, dtype=np.int32).reshape((-1, 1, 2))
        
        # Simplify the path to get key corners
        epsilon = 0.02 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        
        if len(approx) < 2:
            continue
            
        pts = [p[0] for p in approx]
        
        # Rectilinearization: Force each segment to be H or V
        rect_pts = [pts[0]]
        for j in range(1, len(pts)):
            prev = rect_pts[-1]
            curr = pts[j]
            
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            
            if dx > dy:
                # Horizontal: Use current X, keep previous Y
                rect_pts.append([curr[0], prev[1]])
            else:
                # Vertical: Keep previous X, use current Y
                rect_pts.append([prev[0], curr[1]])
        
        # Draw the segments
        for k in range(len(rect_pts) - 1):
            p1 = tuple(map(int, rect_pts[k]))
            p2 = tuple(map(int, rect_pts[k+1]))
            cv2.line(line_img, p1, p2, (0, 255, 0), 1)
            
    return line_img, skeleton

def test_on_image(image_path, output_path):
    print(f"Processing: {image_path}")
    
    # 1. Get cropped image (matching arrow_graph_converter logic)
    img_gray = crop_central_arrows(image_path)
    if img_gray is None:
        print("Failed to load/crop image.")
        return

    # 2. Threshold to get binary
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")
    
    # 4. Draw Midlines
    # We pass the shape of the original image
    result_lines, raw_skeleton = draw_midlines_with_l_turns(contours, img_gray.shape)
    
    # 5. Merge with original for visualization
    # Convert gray to BGR
    if len(img_gray.shape) == 2:
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_gray.copy()
        
    # Overlay lines (green) on original
    # Create mask of lines
    lines_gray = cv2.cvtColor(result_lines, cv2.COLOR_BGR2GRAY)
    _, lines_mask = cv2.threshold(lines_gray, 10, 255, cv2.THRESH_BINARY)
    
    # Paint green where lines are
    img_color[lines_mask > 0] = (0, 255, 0) # Green lines
    
    # Also save the raw skeleton for debugging
    cv2.imwrite(output_path.replace(".png", "_skeleton.png"), raw_skeleton)
    cv2.imwrite(output_path, img_color)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    input_file = os.path.join(project_root, "sample_ui", "level2.jpg")
    output_file = os.path.join(base_dir, "midlines_level2.png")
    
    if os.path.exists(input_file):
        test_on_image(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
