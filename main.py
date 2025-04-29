import logging
from image_processing import read_pdf, convert_pdf_pages_to_images, split_spreads_to_pages

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# INPUT_PDF = "CCAG01.pdf"

# # Example usage
# if __name__ == "__main__":
#     logging.info("Starting PDF processing")
#     pdf_reader = read_pdf(INPUT_PDF)
    
#     # First convert PDF to spread images
#     spread_images = convert_pdf_pages_to_images(
#         INPUT_PDF,
#         page_range=(1, len(pdf_reader.pages))
#     )
    
#     # Then split spreads into individual pages with debug output
#     page_images = split_spreads_to_pages(spread_images)
    
#     logging.info("PDF processing complete")

