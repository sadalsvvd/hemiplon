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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
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
            self.project.input_file, page_range=(1, len(pdf_reader.pages))
        )

        # Split spreads into individual pages
        page_images = split_spreads_to_pages(spread_images)

        # Move images to project directory
        for img in page_images:
            shutil.move(img, self.project.images_dir / Path(img).name)

        logger.info(f"PDF processing complete for {self.project.name}")

    async def run_transcription(
        self, start_index: int = 0, end_index: int | None = None
    ) -> None:
        """
        Run transcription for all configured models and runs.

        Args:
            start_index: Index of first image to process (inclusive)
            end_index: Index of last image to process (exclusive). If None, process all remaining images.
        """
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""

                logger.info(f"Running transcription for {config.model} (run {run + 1})")

                await process_directory(
                    directory_path=str(self.project.images_dir),
                    ocr_prompt_path=config.ocr_prompt_path,
                    max_concurrent=config.max_concurrent,
                    start_index=start_index,
                    end_index=end_index,
                    outpath_postfix=f"_{config.model}{run_suffix}",
                    model=config.model,
                    output_dir=str(self.project.output_dir),
                )

    def generate_diffs(self) -> None:
        """Generate diffs between different transcription runs and write each to its own file."""
        # Get all transcription output directories
        transcription_dirs = []
        labels = []

        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""
                dir_name = f"transcribed_{config.model}{run_suffix}"
                dir_path = self.project.output_dir / dir_name

                # Read run metadata
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_name)

                transcription_dirs.append(str(dir_path))

        # Generate multi-way diffs
        diffs = compare_multiple_folders(
            folders=transcription_dirs, labels=labels, file_pattern="*.md"
        )

        # Write each diff to its own file in output/diffs
        diffs_dir = self.project.output_dir / "diffs"
        diffs_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.project.name  # e.g. CCAG01
        for filename, diff_text in diffs.items():
            # filename is like CCAG01_page_0001_transcribed.md
            # Replace _transcribed.md with _diff.md
            if filename.endswith("_transcribed.md"):
                diff_filename = filename.replace("_transcribed.md", "_diff.md")
            else:
                # fallback: just append _diff.md
                base = filename.rsplit(".", 1)[0]
                diff_filename = f"{base}_diff.md"
            diff_path = diffs_dir / diff_filename
            with open(diff_path, "w", encoding="utf-8") as f:
                f.write(diff_text)
            logger.info(f"Wrote diff to {diff_path}")

    async def review_transcriptions(self) -> None:
        """Review transcription differences and generate assessment for each diff file."""
        logger.info(f"Starting transcription review for project {self.project.name}")

        # Check if we have review configuration
        if not self.project.transcription_review:
            logger.warning("No transcription review configuration found")
            return

        # Directory containing per-page diffs
        diffs_dir = self.project.output_dir / "diffs"
        if not diffs_dir.exists():
            logger.error("No diffs directory found. Run generate_diffs first.")
            return

        # Output directory for reviewed transcriptions
        reviewed_dir = self.project.output_dir / "transcribed_reviewed"
        reviewed_dir.mkdir(parents=True, exist_ok=True)

        # Read review prompt
        with open(self.project.review_prompt_path, "r") as f:
            review_prompt = f.read()

        # For each diff file
        for diff_file in sorted(diffs_dir.glob("*_diff.md")):
            with open(diff_file, "r") as f:
                diffs_content = f.read()

            for config in self.project.transcription_review:
                logger.info(f"Reviewing {diff_file.name} with model {config.model}")
                try:
                    review_completion = self.client.chat.completions.create(
                        model=config.model,
                        messages=[
                            {"role": "system", "content": review_prompt},
                            {"role": "user", "content": diffs_content},
                        ],
                        max_tokens=24000,
                    )
                    review_text = review_completion.choices[0].message.content
                    assert review_text is not None, "Review text is None"
                    # Output file: CCAG01_page_####_transcribed_reviewed.md
                    base_name = diff_file.name.replace(
                        "_diff.md", "_transcribed_reviewed.md"
                    )
                    review_path = reviewed_dir / base_name
                    with open(review_path, "w") as out_f:
                        out_f.write(review_text)
                    logger.info(f"Review for {diff_file.name} saved to {review_path}")
                except Exception as e:
                    logger.error(
                        f"Error during review of {diff_file.name} with {config.model}: {str(e)}"
                    )
                    raise

    async def finalize_transcriptions(self) -> None:
        """Generate a final unified transcription for each page using all originals and the review rationale. If the review is 'Consensus.', just use the first available original."""
        logger.info(
            f"Starting transcription finalization for project {self.project.name}"
        )

        # Directory containing per-page reviewed summaries
        reviewed_dir = self.project.output_dir / "transcribed_reviewed"
        if not reviewed_dir.exists():
            logger.error(
                "No reviewed directory found. Run review_transcriptions first."
            )
            return

        # Directory containing all original transcriptions
        transcription_dirs = []
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""
                dir_name = f"transcribed_{config.model}{run_suffix}"
                dir_path = self.project.output_dir / dir_name
                transcription_dirs.append(dir_path)

        # Output directory for final transcriptions
        final_dir = self.project.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # For each reviewed file
        for review_file in sorted(reviewed_dir.glob("*_transcribed_reviewed.md")):
            page_id = review_file.name.replace("_transcribed_reviewed.md", "")
            # Gather all original transcriptions for this page
            originals = {}
            for tdir in transcription_dirs:
                # Find the matching file in this dir
                candidates = list(tdir.glob(f"{page_id}_transcribed.md"))
                if candidates:
                    # Use the directory name as the label
                    label = tdir.name
                    with open(candidates[0], "r") as f:
                        originals[label] = f.read()
            if not originals:
                # TODO: WHY IS THIS HAPPENING?
                logger.warning(f"No originals found for {page_id}")
                continue
            # Read the review rationale
            with open(review_file, "r") as f:
                review_text = f.read().strip()
            # If consensus, just use the first available original
            if review_text == "Consensus.":
                first_label = next(iter(originals))
                final_text = originals[first_label]
                logger.info(
                    f"Consensus for {page_id}: using {first_label} as final output."
                )
            else:
                # Compose the prompt
                prompt = (
                    "You are a scholarly assistant. Here are the original transcriptions from different models/runs, "
                    "and a review summary of their differences and recommendations. "
                    "Using both, produce the best possible unified transcription for this page.\n\n"
                    "# Originals:\n"
                )
                for label, text in originals.items():
                    prompt += f"## {label}\n{text}\n\n"
                prompt += (
                    "# Review Summary:\n"
                    + review_text
                    + "\n\n# Final Unified Transcription (just the text, no commentary):\n"
                )
                # Use the first review model for finalization
                config = self.project.transcription_review[0]
                try:
                    completion = self.client.chat.completions.create(
                        model=config.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=24000,
                    )
                    final_text = completion.choices[0].message.content
                    assert final_text is not None, "Final text is None"
                except Exception as e:
                    logger.error(f"Error during finalization for {page_id}: {str(e)}")
                    raise
            final_path = final_dir / f"{page_id}_final.md"
            with open(final_path, "w") as out_f:
                out_f.write(final_text)
            logger.info(f"Final transcription for {page_id} saved to {final_path}")

    async def run_pipeline(
        self,
        stages: List[str] | None = None,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> None:
        """
        Run the complete project pipeline or specific stages.

        Args:
            stages: List of stages to run. If None, runs all stages.
                   Valid stages: ['pdf', 'transcription', 'transcription-diff', 'transcription-review', 'finalize']
            start_index: For transcription stage, index of first image to process (inclusive)
            end_index: For transcription stage, index of last image to process (exclusive)
        """
        logger.info(f"Starting pipeline for project {self.project.name}")

        # If no stages specified, run all
        if stages is None:
            stages = [
                "pdf",
                "transcription",
                "transcription-diff",
                "transcription-review",
                "transcription-finalize",
            ]

        # Process PDF
        if "pdf" in stages:
            await self.process_pdf()

        # Run transcriptions
        if "transcription" in stages:
            await self.run_transcription(start_index=start_index, end_index=end_index)

        # Generate diffs
        if "transcription-diff" in stages:
            self.generate_diffs()

        # Run review
        if "transcription-review" in stages:
            await self.review_transcriptions()

        # Run finalize
        if "transcription-finalize" in stages:
            await self.finalize_transcriptions()

        logger.info(f"Pipeline complete for project {self.project.name}")


def create_project(name: str, input_file: str) -> Project:
    """Create a new project with default configuration"""
    return Project.create(name, input_file)


def load_project(name: str) -> Project:
    """Load an existing project"""
    return Project.from_yaml(name)
