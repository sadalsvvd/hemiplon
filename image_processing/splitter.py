import cv2
import numpy as np
import os

def split_two_page(image_path, padding=5, invert_for_dark_line=True, debug=False):
    """
    Split a two-page spread into individual pages.
    This is now just a wrapper around split_by_hough which is more reliable for detecting
    the strong vertical gutter line.
    
    Args:
        image_path: Path to the input image
        padding: Number of pixels to pad around the split line
        invert_for_dark_line: Whether to invert the image for dark gutter lines
        debug: If True, saves intermediate images to debug directory
    """
    return split_by_hough(image_path, padding, debug)

def split_by_hough(image_path, padding=5, debug=False):
    """
    Split a two-page spread into individual pages using Hough line detection.
    Specifically looks for the strong vertical line in the gutter between pages.
    
    Args:
        image_path: Path to the input image
        padding: Number of pixels to pad around the split line
        debug: If True, saves intermediate images to debug directory
    """
    # Initialize debug variables
    debug_dir = ""
    base_name = ""
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_1_grayscale.jpg"), gray)

    # Use Canny edge detection with tight thresholds to find strong edges
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_2_edges.jpg"), edges)

    # Detect lines using Hough transform
    # Increase threshold to only detect strong lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                           minLineLength=img.shape[0] * 0.5,  # At least 50% of image height
                           maxLineGap=20)

    if lines is None:
        # Fallback to center if no lines detected
        split_x = img.shape[1] // 2
    else:
        # Filter for vertical lines and find the most central one
        center_x = img.shape[1] // 2
        best_x = center_x
        min_dist_to_center = float('inf')
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is vertical (small x difference)
            if abs(x1 - x2) < 20:
                # Calculate average x position of the line
                avg_x = (x1 + x2) // 2
                # Calculate height of the line
                height = abs(y2 - y1)
                # Check if line is tall enough (at least 50% of image height)
                if height > img.shape[0] * 0.5:
                    # Find the line closest to center
                    dist_to_center = abs(avg_x - center_x)
                    if dist_to_center < min_dist_to_center:
                        min_dist_to_center = dist_to_center
                        best_x = avg_x
        
        split_x = best_x

    if debug:
        # Draw the split line on the image
        marked = img.copy()
        cv2.line(marked, (split_x, 0), (split_x, img.shape[0]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_3_split.jpg"), marked)

    # Split the image
    left = img[:, :split_x - padding]
    right = img[:, split_x + padding:]

    return left, right


# # usage:
# left_page, right_page = split_by_hough(
#     'images/CCAG01_page_0010.jpg'
# )
# cv2.imwrite('hough_left.jpg',  left_page)
# cv2.imwrite('hough_right.jpg', right_page)
