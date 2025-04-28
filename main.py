import os
import PyPDF2
from PIL import Image
import pdf2image
import logging

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
    Convert PDF pages to images and save them in the 'images' folder, with an optional range of pages.
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

    if not os.path.exists("images"):
        os.makedirs("images")

    converted_images = []
    for idx, img in enumerate(pages):
        page_num = idx + page_offset
        out_path = f"images/{pdf_filename}_page_{page_num:04d}.jpg"
        img.save(out_path, "JPEG")
        converted_images.append(out_path)

    logging.info(f"Converted {len(converted_images)} pages to images")
    return converted_images


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

INPUT_PDF = "CCAG01.pdf"

# Example usage
if __name__ == "__main__":
    logging.info("Starting PDF processing")
    pdf_reader = read_pdf(INPUT_PDF)
    images = convert_pdf_pages_to_images(
        INPUT_PDF,
        page_range=(1, len(pdf_reader.pages))
    )

