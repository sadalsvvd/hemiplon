import logging
from typing import List
import asyncio
import os
from dotenv import load_dotenv
import yaml

from .project import Project
from lib.transcribe import process_directory
from utils.diff_checker import compare_multiple_folders
from lib.pdf_processor import PDFProcessor
from lib.llm_service import LLMService
from lib.file_manager import FileManager

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
        self.llm_service = LLMService()

    async def process_pdf(self) -> None:
        """Convert PDF to individual page images"""
        logger.info(f"Processing PDF for project {self.project.name}")

        processor = PDFProcessor(self.project)
        page_images = processor.process_pdf()
        logger.info(f"PDF processing complete for {self.project.name} ({len(page_images)} pages)")

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
                    output_dir=str(self.project.transcription_dir),
                )

    def _get_transcription_dirs(self, as_str=True):
        dirs = []
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""
                dir_name = f"transcribed_{config.model}{run_suffix}"
                dir_path = self.project.transcription_dir / dir_name
                dirs.append(str(dir_path) if as_str else dir_path)
        return dirs

    def generate_diffs(self) -> None:
        """Generate diffs between different transcription runs and write each to its own file."""
        transcription_dirs = self._get_transcription_dirs(as_str=True)
        labels = []
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""
                dir_name = f"transcribed_{config.model}{run_suffix}"
                dir_path = self.project.transcription_dir / dir_name
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_name)
        diffs = compare_multiple_folders(
            folders=transcription_dirs, labels=labels, file_pattern="*.md"
        )
        diffs_dir = self.project.transcription_dir / "diffs"
        FileManager.ensure_dir(diffs_dir)
        prefix = self.project.name  # e.g. CCAG01
        for filename, diff_text in diffs.items():
            if filename.endswith("_transcribed.md"):
                diff_filename = filename.replace("_transcribed.md", "_diff.md")
            else:
                base = filename.rsplit(".", 1)[0]
                diff_filename = f"{base}_diff.md"
            diff_path = diffs_dir / diff_filename
            FileManager.write_text(diff_path, diff_text)
            logger.info(f"Wrote diff to {diff_path}")

    async def review_transcriptions(self) -> None:
        """Review transcription differences and generate assessment for each diff file."""
        logger.info(f"Starting transcription review for project {self.project.name}")
        if not self.project.transcription_review:
            logger.warning("No transcription review configuration found")
            return
        diffs_dir = self.project.transcription_dir / "diffs"
        if not diffs_dir.exists():
            logger.error("No diffs directory found. Run generate_diffs first.")
            return
        reviewed_dir = self.project.transcription_dir / "transcribed_reviewed"
        FileManager.ensure_dir(reviewed_dir)
        with open(self.project.review_prompt_path, "r") as f:
            review_prompt = f.read()
        tasks = []
        for diff_file in sorted(diffs_dir.glob("*_diff.md")):
            diffs_content = FileManager.read_text(diff_file)
            for config in self.project.transcription_review:
                tasks.append(self._review_transcription_task(diff_file, config, review_prompt, diffs_content, reviewed_dir))
        await asyncio.gather(*tasks)

    async def _review_transcription_task(self, diff_file, config, review_prompt, diffs_content, reviewed_dir):
        async with self.llm_service.semaphore:
            logger.info(f"Reviewing {diff_file.name} with model {config.model}")
            try:
                review_text = await self.llm_service.chat(
                    model=config.model,
                    messages=[{"role": "system", "content": review_prompt}, {"role": "user", "content": diffs_content}],
                )
                assert review_text is not None, "Review text is None"
                base_name = diff_file.name.replace("_diff.md", "_transcribed_reviewed.md")
                review_path = reviewed_dir / base_name
                FileManager.write_text(review_path, review_text)
                logger.info(f"Review for {diff_file.name} saved to {review_path}")
            except Exception as e:
                logger.error(f"Error during review of {diff_file.name} with {config.model}: {str(e)}")
                raise

    async def finalize_transcriptions(self) -> None:
        """Generate a final unified transcription for each page using all originals and the review rationale. If the review is 'Consensus.', just use the first available original."""
        logger.info(
            f"Starting transcription finalization for project {self.project.name}"
        )
        reviewed_dir = self.project.transcription_dir / "transcribed_reviewed"
        if not reviewed_dir.exists():
            logger.error("No reviewed directory found. Run review_transcriptions first.")
            return
        transcription_dirs = self._get_transcription_dirs(as_str=False)
        final_dir = self.project.transcription_dir / "final"
        FileManager.ensure_dir(final_dir)
        tasks = []
        for review_file in sorted(reviewed_dir.glob("*_transcribed_reviewed.md")):
            tasks.append(self._finalize_transcription_task(review_file, transcription_dirs, final_dir))
        await asyncio.gather(*tasks)

    async def _finalize_transcription_task(self, review_file, transcription_dirs, final_dir):
        page_id = review_file.name.replace("_transcribed_reviewed.md", "")
        originals = {}
        for tdir in transcription_dirs:
            candidates = list(tdir.glob(f"{page_id}_transcribed.md"))
            if candidates:
                label = tdir.name
                originals[label] = FileManager.read_text(candidates[0])
        if not originals:
            logger.warning(f"No originals found for {page_id}")
            return
        review_text = FileManager.read_text(review_file).strip()
        if review_text == "Consensus.":
            final_text = originals[next(iter(originals))]
            logger.info(f"Consensus for {page_id}: using {next(iter(originals))} as final output.")
        else:
            originals_block = "".join([f"## {label}\n{text}\n\n" for label, text in originals.items()])
            prompt = self.llm_service.render_prompt(
                "prompts/finalize_transcription.j2",
                {"originals_block": originals_block, "review_summary": review_text}
            )
            config = self.project.transcription_review[0]
            try:
                final_text = await self.llm_service.chat(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                assert final_text is not None, "Final text is None"
            except Exception as e:
                logger.error(f"Error during finalization for {page_id}: {str(e)}")
                return
        final_path = final_dir / f"{page_id}_final.md"
        FileManager.write_text(final_path, final_text)
        logger.info(f"Final transcription for {page_id} saved to {final_path}")

    async def run_translation(self) -> None:
        """Run translation for all finalized transcriptions using configured models and runs. Supports previous_source_text for context."""
        logger.info(f"Starting translation for project {self.project.name}")
        final_dir = self.project.transcription_dir / "final"
        if not final_dir.exists():
            logger.error("No finalized transcriptions found. Run finalize_transcriptions first.")
            return
        final_files = sorted(final_dir.glob("*_final.md"))
        page_ids = [f.stem.replace("_final", "") for f in final_files]
        page_id_to_file = {pid: f for pid, f in zip(page_ids, final_files)}
        tasks = []
        for config in self.project.translation:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                out_dir = self.project.translation_dir / f"translated_{config.model}{run_suffix}"
                FileManager.ensure_dir(out_dir)
                for idx, page_id in enumerate(page_ids):
                    tasks.append(self._run_translation_task(config, page_id, idx, page_ids, page_id_to_file, out_dir))
        await asyncio.gather(*tasks)

    async def _run_translation_task(self, config, page_id, idx, page_ids, page_id_to_file, out_dir):
        final_file = page_id_to_file[page_id]
        final_text = FileManager.read_text(final_file)
        previous_source_text = None
        if idx > 0:
            prev_page_id = page_ids[idx-1]
            prev_file = page_id_to_file[prev_page_id]
            previous_source_text = FileManager.read_text(prev_file)
        prompt = self.llm_service.render_prompt(
            "prompts/translate.j2",
            {"source_text": final_text, "previous_source_text": previous_source_text}
        )
        try:
            translation = await self.llm_service.chat(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
            )
            assert translation is not None, "Translation is None"
            out_path = out_dir / f"{page_id}_translated.md"
            FileManager.write_text(out_path, translation)
            logger.info(f"Translation for {page_id} saved to {out_path}")
        except Exception as e:
            logger.error(f"Error during translation for {page_id}: {str(e)}")
            return

    def _get_translation_dirs(self, as_str=True):
        dirs = []
        for config in self.project.translation:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                dir_name = f"translated_{config.model}{run_suffix}"
                dir_path = self.project.translation_dir / dir_name
                dirs.append(str(dir_path) if as_str else dir_path)
        return dirs

    def generate_translation_diffs(self) -> None:
        """Generate diffs between different translation runs and write each to its own file."""
        logger.info(f"Generating translation diffs for project {self.project.name}")
        translation_dirs = self._get_translation_dirs(as_str=True)
        labels = []
        for config in self.project.translation:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                dir_name = f"translated_{config.model}{run_suffix}"
                dir_path = self.project.translation_dir / dir_name
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_name)
        diffs = compare_multiple_folders(
            folders=translation_dirs, labels=labels, file_pattern="*.md"
        )
        diffs_dir = self.project.translation_dir / "diffs"
        FileManager.ensure_dir(diffs_dir)
        for filename, diff_text in diffs.items():
            if filename.endswith("_translated.md"):
                diff_filename = filename.replace("_translated.md", "_diff.md")
            else:
                base = filename.rsplit(".", 1)[0]
                diff_filename = f"{base}_diff.md"
            diff_path = diffs_dir / diff_filename
            FileManager.write_text(diff_path, diff_text)
            logger.info(f"Wrote translation diff to {diff_path}")

    async def review_translations(self) -> None:
        """Review translation differences and generate assessment for each diff file using the translation_review.j2 template."""
        logger.info(f"Starting translation review for project {self.project.name}")
        diffs_dir = self.project.translation_dir / "diffs"
        if not diffs_dir.exists():
            logger.error("No translation diffs directory found. Run generate_translation_diffs first.")
            return
        reviewed_dir = self.project.translation_dir / "reviewed"
        FileManager.ensure_dir(reviewed_dir)
        tasks = []
        for diff_file in sorted(diffs_dir.glob("*_diff.md")):
            diffs_content = FileManager.read_text(diff_file)
            review_prompt = self.llm_service.render_prompt(
                "prompts/translation_review.j2",
                {"diff_content": diffs_content}
            )
            for config in self.project.translation:
                tasks.append(self._review_translation_task(diff_file, config, review_prompt, reviewed_dir))
        await asyncio.gather(*tasks)

    async def _review_translation_task(self, diff_file, config, review_prompt, reviewed_dir):
        async with self.llm_service.semaphore:
            logger.info(f"Reviewing translation {diff_file.name} with model {config.model}")
            try:
                review_text = await self.llm_service.chat(
                    model=config.model,
                    messages=[{"role": "system", "content": review_prompt}],
                )
                assert review_text is not None, "Review text is None"
                base_name = diff_file.name.replace("_diff.md", "_reviewed.md")
                review_path = reviewed_dir / base_name
                FileManager.write_text(review_path, review_text)
                logger.info(f"Translation review for {diff_file.name} saved to {review_path}")
            except Exception as e:
                logger.error(f"Error during translation review of {diff_file.name} with {config.model}: {str(e)}")
                return

    async def finalize_translations(self) -> None:
        """Generate a final unified translation for each page using all originals and the review rationale. If the review is 'Consensus.', just use the first available translation, or fall back to the finalized transcription."""
        logger.info(f"Starting translation finalization for project {self.project.name}")
        reviewed_dir = self.project.translation_dir / "reviewed"
        if not reviewed_dir.exists():
            logger.error("No reviewed translation directory found. Run review_translations first.")
            return
        translation_dirs = self._get_translation_dirs(as_str=False)
        final_dir = self.project.translation_dir / "final"
        FileManager.ensure_dir(final_dir)
        tasks = []
        for review_file in sorted(reviewed_dir.glob("*_reviewed.md")):
            tasks.append(self._finalize_translation_task(review_file, translation_dirs, final_dir))
        await asyncio.gather(*tasks)

    async def _finalize_translation_task(self, review_file, translation_dirs, final_dir):
        page_id = review_file.name.replace("_reviewed.md", "")
        originals = {}
        for tdir in translation_dirs:
            candidates = list(tdir.glob(f"{page_id}_translated.md"))
            if candidates:
                label = tdir.name
                originals[label] = FileManager.read_text(candidates[0])
        review_text = FileManager.read_text(review_file).strip()
        if review_text == "Consensus.":
            if originals:
                final_text = originals[next(iter(originals))]
                logger.info(f"Consensus for {page_id}: using {next(iter(originals))} as final translation.")
            else:
                final_trans_path = self.project.transcription_dir / "final" / f"{page_id}_final.md"
                if final_trans_path.exists():
                    final_text = FileManager.read_text(final_trans_path)
                    logger.info(f"Consensus for {page_id}: using finalized transcription as final translation.")
                else:
                    logger.warning(f"Consensus for {page_id}: no translation or finalized transcription found.")
                    return
        else:
            originals_block = "".join([f"## {label}\n{text}\n\n" for label, text in originals.items()])
            prompt = self.llm_service.render_prompt(
                "prompts/finalize_translation.j2",
                {"originals_block": originals_block, "review_summary": review_text}
            )
            config = self.project.translation[0]
            try:
                final_text = await self.llm_service.chat(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                assert final_text is not None, "Final translation is None"
            except Exception as e:
                logger.error(f"Error during translation finalization for {page_id}: {str(e)}")
                return
        final_path = final_dir / f"{page_id}_final.md"
        FileManager.write_text(final_path, final_text)
        logger.info(f"Final translation for {page_id} saved to {final_path}")

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
                   Valid stages: ['pdf', 'transcription', 'transcription-diff', 'transcription-review', 'transcription-finalize', 'translation', 'translation-diff', 'translation-review', 'translation-finalize']
            start_index: For transcription stage, index of first image to process (inclusive)
            end_index: For transcription stage, index of last image to process (exclusive)
        """
        logger.info(f"Starting pipeline for project {self.project.name}")
        
        if stages is None:
            stages = [
                "pdf",
                "transcription",
                "transcription-diff",
                "transcription-review",
                "transcription-finalize",
                "translation",
                "translation-diff",
                "translation-review",
                "translation-finalize",
            ]
        if "pdf" in stages:
            await self.process_pdf()
        if "transcription" in stages:
            await self.run_transcription(start_index=start_index, end_index=end_index)
        if "transcription-diff" in stages:
            self.generate_diffs()
        if "transcription-review" in stages:
            await self.review_transcriptions()
        if "transcription-finalize" in stages:
            await self.finalize_transcriptions()
        if "translation" in stages:
            await self.run_translation()
        if "translation-diff" in stages:
            self.generate_translation_diffs()
        if "translation-review" in stages:
            await self.review_translations()
        if "translation-finalize" in stages:
            await self.finalize_translations()
        logger.info(f"Pipeline complete for project {self.project.name}")


def create_project(name: str, input_file: str) -> Project:
    """Create a new project with default configuration"""
    return Project.create(name, input_file)


def load_project(name: str) -> Project:
    """Load an existing project"""
    return Project.from_yaml(name)
