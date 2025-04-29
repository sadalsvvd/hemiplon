import logging
from pathlib import Path
import shutil
from typing import List
import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml

from .project import Project
from lib.pdf import read_pdf, convert_pdf_pages_to_images, split_spreads_to_pages
from lib.transcribe import process_directory, encode_image, write_transcription
from utils.diff_checker import compare_multiple_folders, save_diffs_to_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assert api_key, "OPENAI_API_KEY is not set"

class ProjectManager:
    def __init__(self, project: Project):
        self.project = project
        self.client = OpenAI(api_key=api_key)
    
    async def process_pdf(self) -> None:
        """Convert PDF to individual page images"""
        logger.info(f"Processing PDF for project {self.project.name}")
        
        # Read PDF
        pdf_reader = read_pdf(self.project.input_file)
        
        # Convert to spread images
        spread_images = convert_pdf_pages_to_images(
            self.project.input_file,
            page_range=(1, len(pdf_reader.pages))
        )
        
        # Split spreads into individual pages
        page_images = split_spreads_to_pages(spread_images)
        
        # Move images to project directory
        for img in page_images:
            shutil.move(img, self.project.images_dir / Path(img).name)
        
        logger.info(f"PDF processing complete for {self.project.name}")
    
    async def run_transcription(self, start_index: int = 0, end_index: int | None = None) -> None:
        """
        Run transcription for all configured models and runs.
        
        Args:
            start_index: Index of first image to process (inclusive)
            end_index: Index of last image to process (exclusive). If None, process all remaining images.
        """
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                
                logger.info(f"Running transcription for {config.model} (run {run+1})")
                
                await process_directory(
                    directory_path=str(self.project.images_dir),
                    ocr_prompt_path=config.ocr_prompt_path,
                    max_concurrent=config.max_concurrent,
                    start_index=start_index,
                    end_index=end_index,
                    outpath_postfix=f"_{config.model}{run_suffix}",
                    model=config.model,
                    output_dir=str(self.project.output_dir)
                )
    
    def generate_diffs(self) -> None:
        """Generate diffs between different transcription runs"""
        # Get all transcription output directories
        transcription_dirs = []
        labels = []
        
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                dir_name = f"transcribed_{config.model}{run_suffix}"
                dir_path = self.project.output_dir / dir_name
                
                # Read run metadata
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_name)
                
                transcription_dirs.append(str(dir_path))
        
        # Generate multi-way diffs
        diffs = compare_multiple_folders(
            folders=transcription_dirs,
            labels=labels,
            file_pattern="*.md"
        )
        
        # Save diffs
        save_diffs_to_file(
            diffs,
            str(self.project.output_dir / "transcription_diffs.txt")
        )
    
    async def run_pipeline(self, stages: List[str] | None = None, start_index: int = 0, end_index: int | None = None) -> None:
        """
        Run the complete project pipeline or specific stages.
        
        Args:
            stages: List of stages to run. If None, runs all stages.
                   Valid stages: ['pdf', 'transcription', 'diffs']
            start_index: For transcription stage, index of first image to process (inclusive)
            end_index: For transcription stage, index of last image to process (exclusive)
        """
        logger.info(f"Starting pipeline for project {self.project.name}")
        
        # If no stages specified, run all
        if stages is None:
            stages = ['pdf', 'transcription', 'diffs']
        
        # Process PDF
        if 'pdf' in stages:
            await self.process_pdf()
        
        # Run transcriptions
        if 'transcription' in stages:
            await self.run_transcription(start_index=start_index, end_index=end_index)
        
        # Generate diffs
        if 'diffs' in stages:
            self.generate_diffs()
        
        logger.info(f"Pipeline complete for project {self.project.name}")

def create_project(name: str, input_file: str) -> Project:
    """Create a new project with default configuration"""
    return Project.create(name, input_file)

def load_project(name: str) -> Project:
    """Load an existing project"""
    return Project.from_yaml(name) 