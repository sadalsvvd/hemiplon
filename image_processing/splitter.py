import cv2
import numpy as np

def split_two_page(image_path, padding=10, invert_for_dark_line=True):
    # 1) Load and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) (Optional) invert if your gutter is a dark rule on a light page
    if invert_for_dark_line:
        gray = 255 - gray

    # 3) Threshold to binary so the gutter line stands out
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 4) Sum up “ink” in each column
    col_sums = np.sum(bw == 0, axis=0)  # count black pixels per column

    # 5) Smooth the signal a bit
    kernel = np.ones(50, dtype=np.int32)
    smooth = np.convolve(col_sums, kernel, mode='same')

    # 6) Look for the peak nearest the image-center
    center = smooth.shape[0] // 2
    # search window of ±20% width around center to avoid stray margins
    window = int(smooth.shape[0] * 0.2)
    search = smooth[center - window : center + window]
    peak = np.argmax(search) + (center - window)

    # 7) Compute crop boundaries
    h, w = gray.shape
    x = int(peak)
    left = img[:, : x - padding]
    right = img[:, x + padding :]

    return left, right

# usage:
left_page, right_page = split_two_page(
    'images/CCAG01_page_0010.jpg'
)
cv2.imwrite('left.jpg',  left_page)
cv2.imwrite('right.jpg', right_page)
