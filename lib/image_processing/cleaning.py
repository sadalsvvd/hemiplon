import cv2
import numpy as np
import os
from typing import Optional

def clean_small_artifacts(image_path: str, min_component_size: int = 10, debug: bool = False):
    """
    Clean small dots and artifacts from document images while preserving text.
    Uses connected component analysis to identify and remove components smaller
    than a threshold size.
    
    Args:
        image_path: Path to input image
        min_component_size: Minimum size in pixels for a component to be kept
        debug: If True, saves debug visualizations
        
    Returns:
        Cleaned image with small artifacts removed
    """
    # Read image and convert to grayscale if needed
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Initialize debug variables
    debug_dir: Optional[str] = None
    base_name: Optional[str] = None
    
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        assert debug_dir is not None and base_name is not None
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_1_original.jpg"), gray)

    # Threshold to binary - use Otsu's method for automatic threshold selection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if debug and debug_dir is not None and base_name is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_2_binary.jpg"), binary)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    cleaned = np.zeros_like(binary)
    
    # Copy only components larger than min_component_size
    # Skip label 0 which is the background
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned[labels == label] = 255
            
    if debug and debug_dir is not None and base_name is not None:
        # Visualize removed components in red
        debug_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        removed_mask = (binary > 0) & (cleaned == 0)
        debug_vis[removed_mask] = [0, 0, 255]  # Red
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_3_removed.jpg"), debug_vis)
    
    # Invert back to black text on white background
    cleaned = cv2.bitwise_not(cleaned)
    
    if debug and debug_dir is not None and base_name is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_4_final.jpg"), cleaned)
    
    return cleaned 