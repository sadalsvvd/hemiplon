import os
import PyPDF2
from PIL import Image
import pdf2image
import logging
from .splitter import split_two_page
from .cleaning import clean_small_artifacts
import cv2

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_pdf(file_path):
    """
    Read in a PDF file
    """
    logging.info(f"Reading PDF file from {file_path}")
    pdf_file = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    logging.info(f"Successfully read PDF file with {len(pdf_reader.pages)} pages")
    return pdf_reader


def convert_pdf_pages_to_images(pdf_path, page_range=None):
    """
    Convert PDF pages to images and save them in the 'images/spreads' folder, with an optional range of pages.
    The output image files will be prefixed with the PDF filename (sans .pdf).
    """
    logging.info(
        f"Converting PDF pages from {pdf_path} to images, page_range={page_range}"
    )
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    if page_range:
        start_page, end_page = page_range
        logging.info(f"Converting pages {start_page} to {end_page}")
        pages = pdf2image.convert_from_path(
            pdf_path,
            first_page=start_page,
            last_page=end_page,
        )
        page_offset = start_page
    else:
        logging.info("Converting all pages")
        pages = pdf2image.convert_from_path(pdf_path)
        page_offset = 1

    # Create spreads directory if it doesn't exist
    spreads_dir = os.path.join("images", "spreads")
    if not os.path.exists(spreads_dir):
        os.makedirs(spreads_dir)

    converted_images = []
    for idx, img in enumerate(pages):
        page_num = idx + page_offset
        out_path = os.path.join(spreads_dir, f"{pdf_filename}_spread_{page_num:04d}.jpg")
        img.save(out_path, "JPEG")
        converted_images.append(out_path)

    logging.info(f"Converted {len(converted_images)} pages to spread images")
    return converted_images


def split_spreads_to_pages(spread_images, debug=False, clean_artifacts=True):
    """
    Split spread images into individual left and right pages.
    Saves pages to 'images/pages' directory with sequential numbering.
    Returns list of paths to the generated page images.
    
    Args:
        spread_images: List of paths to spread images
        debug: If True, enables debug image output for the splitting process
        clean_artifacts: If True, removes small dots and artifacts from the images
    """
    pages_dir = os.path.join("images", "pages")
    if not os.path.exists(pages_dir):
        os.makedirs(pages_dir)

    page_images = []
    for spread_idx, spread_path in enumerate(spread_images):
        # Extract the prefix from the spread filename (everything before _spread_)
        spread_filename = os.path.basename(spread_path)
        prefix = spread_filename.split("_spread_")[0]
        
        # Split the spread into left and right pages
        left_page, right_page = split_two_page(spread_path, debug=debug)
        
        # Calculate the page numbers (0,1 for first spread, 2,3 for second, etc)
        left_page_num = spread_idx * 2
        right_page_num = left_page_num + 1
        
        # Save left page
        left_path = os.path.join(pages_dir, f"{prefix}_page_{left_page_num:04d}.jpg")
        if debug and clean_artifacts:
            # Save original version with _original suffix
            left_orig_path = os.path.join(pages_dir, f"{prefix}_page_{left_page_num:04d}_original.jpg")
            cv2.imwrite(left_orig_path, left_page)
        cv2.imwrite(left_path, left_page)
        if clean_artifacts:
            cleaned = clean_small_artifacts(left_path, debug=debug)
            cv2.imwrite(left_path, cleaned)
        page_images.append(left_path)
        
        # Save right page
        right_path = os.path.join(pages_dir, f"{prefix}_page_{right_page_num:04d}.jpg")
        if debug and clean_artifacts:
            # Save original version with _original suffix
            right_orig_path = os.path.join(pages_dir, f"{prefix}_page_{right_page_num:04d}_original.jpg")
            cv2.imwrite(right_orig_path, right_page)
        cv2.imwrite(right_path, right_page)
        if clean_artifacts:
            cleaned = clean_small_artifacts(right_path, debug=debug)
            cv2.imwrite(right_path, cleaned)
        page_images.append(right_path)

    logging.info(f"Split {len(spread_images)} spreads into {len(page_images)} individual pages")
    return page_images


def export_images_to_pdf(image_files, output_pdf_path):
    """
    Export a new PDF file containing all the changes which were applied
    """
    logging.info(f"Exporting images to PDF at {output_pdf_path}")
    image_objects = [Image.open(img_file).convert("RGB") for img_file in image_files]
    image_objects[0].save(
        output_pdf_path, save_all=True, append_images=image_objects[1:]
    )
    logging.info(f"Successfully exported {len(image_objects)} images to PDF")
