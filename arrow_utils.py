import cv2
import numpy as np

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle at p2 formed by p1-p2-p3 in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    unit_v1 = v1 / norm1
    unit_v2 = v2 / norm2
    
    dot_product = np.dot(unit_v1, unit_v2)
    # Clamp for floating point errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def get_arrow_head_center(contour):
    """
    Takes an OpenCV contour of an arrow shape and returns the center (x, y) 
    of the arrow head triangle using geometric pattern recognition.
    
    Robust to orientation (CW/CCW) and polygon simplification artifacts.
    Finds a sequence of Convex vertices and identifies the Tip as the sharpest
    valid angle within that sequence.
    
    Args:
        contour: Numpy array of point coordinates (opencv contour).
        
    Returns:
        (x, y): Tuple of integers representing the center of the arrow head.
        None: If the arrow head cannot be reliably detected.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    num_points = len(approx)
    if num_points < 3:
        return None
        
    pts = approx[:, 0, :]
    
    # Determine Orientation
    signed_area = cv2.contourArea(contour, True)
    is_ccw = signed_area > 0
    
    vertex_types = [] # 1 for Convex, -1 for Reflex
    angles = []
    
    for i in range(num_points):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % num_points]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Determine Convexity based on Orientation
        # CCW: Cross > 0 is Convex
        # CW: Cross < 0 is Convex
        if is_ccw:
            v_type = 1 if cross > 0 else -1
        else:
            v_type = 1 if cross < 0 else -1
            
        vertex_types.append(v_type)
        angles.append(calculate_angle(p_prev, p_curr, p_next))

    # Find sequences of Convex vertices
    # We look for a Convex vertex that is likely the Tip.
    # It must be Convex.
    # It is usually part of a run of Convex vertices (Wings + Tip).
    
    candidates = []
    
    for i in range(num_points):
        if vertex_types[i] == 1: # Convex
            angle = angles[i]
            # Filter for reasonable arrow tip angles (e.g., 25 to 110 degrees)
            # Very small angles (< 20) are likely artifacts or shaft ends collapsed.
            if 25 <= angle <= 110:
                candidates.append(i)
                
    if not candidates:
        return None
        
    # Pick the best candidate (minimum angle)
    best_tip_idx = -1
    min_angle = 360
    
    for idx in candidates:
        if angles[idx] < min_angle:
            min_angle = angles[idx]
            best_tip_idx = idx
            
    if best_tip_idx != -1:
        tip = pts[best_tip_idx]
        p_prev = pts[(best_tip_idx - 1) % num_points]
        p_next = pts[(best_tip_idx + 1) % num_points]
        
        # Center of triangle
        center_x = int((tip[0] + p_prev[0] + p_next[0]) / 3)
        center_y = int((tip[1] + p_prev[1] + p_next[1]) / 3)
        
        return (center_x, center_y)

    return None
