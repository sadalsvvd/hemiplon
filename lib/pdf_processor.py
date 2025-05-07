from pathlib import Path
from typing import List
from lib.projects.project import Project
from lib.pdf import convert_pdf_pages_to_images, split_spreads_to_pages

class PDFProcessor:
    def __init__(self, project: Project):
        self.project = project
        self.images_dir = project.images_dir
        self.images_dir.mkdir(exist_ok=True)
        self.spreads_dir = self.images_dir / "spreads"
        self.spreads_dir.mkdir(exist_ok=True)
    
    def process_pdf(self) -> List[Path]:
        """Process PDF into individual page images"""
        pdf_path = Path(self.project.input_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert PDF to images, using project.name as prefix
        spread_images = convert_pdf_pages_to_images(
            str(pdf_path),
            page_range=(1, None),  # Process all pages
            output_dir=str(self.spreads_dir),
            project_name=self.project.name
        )
        
        if self.project.two_page_spread:
            # Split spreads into individual pages, using project.name as prefix
            page_images = split_spreads_to_pages(
                spread_images,
                output_dir=str(self.images_dir),
                project_name=self.project.name
            )
            return [Path(img) for img in page_images]
        else:
            # For non-spread PDFs, the spread images are actually single pages
            return [Path(img) for img in spread_images] 