import cv2
import numpy as np
import os

def split_two_page(image_path, padding=5, invert_for_dark_line=True, debug=False):
    """
    Split a two-page spread into individual pages.
    
    Args:
        image_path: Path to the input image
        padding: Number of pixels to pad around the split line
        invert_for_dark_line: Whether to invert the image for dark gutter lines
        debug: If True, saves intermediate images to debug directory
    """
    # Initialize debug variables
    debug_dir = ""
    base_name = ""
    
    # Create debug directory if needed
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1) Load and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_1_grayscale.jpg"), gray)

    # 2) (Optional) invert if your gutter is a dark rule on a light page
    if invert_for_dark_line:
        gray = 255 - gray
        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_2_inverted.jpg"), gray)

    # 3) Threshold to binary with a higher threshold to reduce noise
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # Increased threshold
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_3_binary.jpg"), bw)

    # 4) Sum up "ink" in each column
    col_sums = np.sum(bw == 0, axis=0)  # count black pixels per column
    if debug:
        # Create visualization of column sums
        vis = np.zeros((100, len(col_sums)), dtype=np.uint8)
        max_sum = np.max(col_sums)
        if max_sum > 0:
            vis = (col_sums * 255 / max_sum).astype(np.uint8)
            vis = np.repeat(vis.reshape(1, -1), 100, axis=0)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_4_column_sums.jpg"), vis)

    # 5) Smooth the signal with a larger kernel
    kernel = np.ones(100, dtype=np.int32)  # Increased kernel size
    smooth = np.convolve(col_sums, kernel, mode='same')
    if debug:
        # Create visualization of smoothed signal
        vis = np.zeros((100, len(smooth)), dtype=np.uint8)
        max_smooth = np.max(smooth)
        if max_smooth > 0:
            vis = (smooth * 255 / max_smooth).astype(np.uint8)
            vis = np.repeat(vis.reshape(1, -1), 100, axis=0)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_5_smoothed.jpg"), vis)

    # 6) Look for the peak nearest the image-center with a narrower search window
    center = smooth.shape[0] // 2
    # Narrow search window to ±10% width around center
    window = int(smooth.shape[0] * 0.1)  # Reduced from 0.2 to 0.1
    search = smooth[center - window : center + window]
    peak = np.argmax(search) + (center - window)

    # Add a check to ensure the peak is reasonable
    # If the peak is too far from center, use the center
    if abs(peak - center) > window:
        peak = center

    if debug:
        # Create visualization with peak marked
        vis = np.zeros((100, len(smooth)), dtype=np.uint8)
        max_smooth = np.max(smooth)
        if max_smooth > 0:
            vis = (smooth * 255 / max_smooth).astype(np.uint8)
            vis = np.repeat(vis.reshape(1, -1), 100, axis=0)
            # Mark the peak with a red line
            vis[:, peak] = 255
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_6_peak.jpg"), vis)

    # 7) Compute crop boundaries
    h, w = gray.shape
    x = int(peak)
    left = img[:, : x - padding]
    right = img[:, x + padding :]
    if debug:
        # Save the split result with the split line marked
        marked = img.copy()
        cv2.line(marked, (x, 0), (x, h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_7_split.jpg"), marked)

    return left, right

def split_by_hough(image_path, padding=5, min_line_length=50, max_line_gap=20):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    print(lines)

    # filter vertical lines (dx small, dy large)
    verts = []
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(x1 - x2) < 10 and abs(y2 - y1) > img.shape[0] * 0.5:
            verts.append((x1, y1, x2, y2))

    # pick the one closest to center
    cx = img.shape[1] // 2
    best = min(verts, key=lambda l: abs((l[0]+l[2])//2 - cx))
    gutter_x = (best[0] + best[2]) // 2

    # split
    left  = img[:, : gutter_x - padding]
    right = img[:, gutter_x + padding :]

    return left, right


# # usage:
# left_page, right_page = split_by_hough(
#     'images/CCAG01_page_0010.jpg'
# )
# cv2.imwrite('hough_left.jpg',  left_page)
# cv2.imwrite('hough_right.jpg', right_page)
