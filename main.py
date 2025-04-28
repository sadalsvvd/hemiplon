import logging
from image_processing import read_pdf, convert_pdf_pages_to_images

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PDF = "CCAG01.pdf"

# Example usage
if __name__ == "__main__":
    logging.info("Starting PDF processing")
    pdf_reader = read_pdf(INPUT_PDF)
    images = convert_pdf_pages_to_images(
        INPUT_PDF,
        page_range=(1, len(pdf_reader.pages))
    )

