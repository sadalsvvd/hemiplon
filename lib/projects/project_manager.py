import logging
from typing import List
import asyncio
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from openai import OpenAI

from .project import Project
from lib.transcribe import transcribe_image
from lib.utils.diff_checker import compare_multiple_folders
from lib.utils.pipeline_utils import log_and_filter_unprocessed, run_tasks_for_files
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
        # The ProjectManager orchestrates the full pipeline for a single project instance.
        self.project = project
        self.llm_service = LLMService()

    async def process_pdf(self) -> None:
        # Converts the project's input PDF into individual page images for downstream processing.
        logger.info(f"Processing PDF for project {self.project.name}")

        processor = PDFProcessor(self.project)
        page_images = processor.process_pdf()
        logger.info(f"PDF processing complete for {self.project.name} ({len(page_images)} pages)")

    async def run_transcription(
        self, start_index: int = 0, end_index: int | None = None
    ) -> None:
        # Runs the transcription stage for all configured models and runs. This is the first LLM-based step.
        images_dir = self.project.images_dir
        image_files = sorted([f for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if end_index is None:
            end_index = len(image_files)
        image_files = image_files[start_index:end_index]
        logger.info(f"Found {len(image_files)} image files in {images_dir}")
        logger.info(f"Processing images from index {start_index} to {end_index}")
        for config in self.project.transcription:
            for run in range(config.runs):
                run_suffix = f"_{run + 1}" if config.runs > 1 else ""
                output_dir = self.project.transcription_dir / f"transcribed_{config.model}{run_suffix}"
                output_dir.mkdir(parents=True, exist_ok=True)
                ocr_prompt_path = config.ocr_prompt_path
                with open(ocr_prompt_path, "r") as file:
                    ocr_prompt_contents = file.read()
                to_process = log_and_filter_unprocessed(
                    input_files=image_files,
                    output_dir=output_dir,
                    output_suffix="_transcribed.md",
                    stage_name=f"Transcription{run_suffix} ({config.model})",
                    logger=logger
                )
                if not to_process:
                    continue
                async def process_and_save(img_path):
                    transcription = await transcribe_image(
                        llm_service=self.llm_service,
                        image_path=str(img_path),
                        ocr_prompt=ocr_prompt_contents,
                        model=config.model
                    )
                    output_path = output_dir / f"{Path(img_path).stem}_transcribed.md"

                    with open(output_path, 'w') as f:
                        f.write(transcription)
                    logger.info(f"Transcription written to {output_path}")

                await run_tasks_for_files(
                    to_process,
                    process_and_save,
                    logger=logger,
                    stage_name=f"Transcription{run_suffix} ({config.model})"
                )

    def _get_transcription_dirs(self, as_str=True):
        # Returns all transcription run directories for this project. Used for diffing and review.
        dirs = []
        for config in self.project.transcription:
            for run in range(config.runs):
                dir_path = self.project.transcription_run_dir(config.model, run + 1)
                dirs.append(str(dir_path) if as_str else dir_path)
        return dirs

    def generate_transcription_diffs(self) -> None:
        # Compares all transcription runs and writes out diffs. This is the basis for review and consensus.
        transcription_dirs = self._get_transcription_dirs(as_str=True)
        labels = []
        for config in self.project.transcription:
            for run in range(config.runs):
                dir_path = self.project.transcription_run_dir(config.model, run + 1)
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_path.name)

        diffs = compare_multiple_folders(
            folders=transcription_dirs, labels=labels, file_pattern="*.md"
        )
        diffs_dir = self.project.transcription_diffs_dir
        FileManager.ensure_dir(diffs_dir)

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
        # This stage uses LLMs to review the diffs between transcription runs, producing rationale for consensus.
        logger.info(f"Starting transcription review for project {self.project.name}")
        if not self.project.transcription_review:
            logger.warning("No transcription review configuration found")
            return

        diffs_dir = self.project.transcription_diffs_dir
        if not diffs_dir.exists():
            logger.error("No diffs directory found. Run generate_diffs first.")
            return

        reviewed_dir = self.project.transcription_reviewed_dir
        FileManager.ensure_dir(reviewed_dir)
        review_prompt = FileManager.read_text(self.project.get_prompt_path("transcription_review"))

        input_files = sorted(diffs_dir.glob("*_diff.md"))
        to_process = log_and_filter_unprocessed(
            input_files=input_files,
            output_dir=reviewed_dir,
            output_suffix="_transcribed_reviewed.md",
            stage_name="Transcription Review",
            logger=logger
        )
        if not to_process:
            return

        async def review_task(diff_file):
            diffs_content = FileManager.read_text(diff_file)
            for config in self.project.transcription_review:
                await self._review_transcription_task(diff_file, config, review_prompt, diffs_content, reviewed_dir)

        await run_tasks_for_files(
            to_process,
            review_task,
            logger=logger,
            stage_name="Transcription Review"
        )

    async def _review_transcription_task(self, diff_file, config, review_prompt, diffs_content, reviewed_dir):
        # This is the core LLM call for reviewing a single diff file. Used by the review_transcriptions stage.
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
        # This stage produces the final, unified transcription for each page, using the review rationale and all originals.
        logger.info(
            f"Starting transcription finalization for project {self.project.name}"
        )
        reviewed_dir = self.project.transcription_reviewed_dir
        if not reviewed_dir.exists():
            logger.error("No reviewed directory found. Run review_transcriptions first.")
            return

        transcription_dirs = self._get_transcription_dirs(as_str=False)
        final_dir = self.project.transcription_final_dir
        FileManager.ensure_dir(final_dir)

        input_files = sorted(reviewed_dir.glob("*_transcribed_reviewed.md"))
        to_process = log_and_filter_unprocessed(
            input_files=input_files,
            output_dir=final_dir,
            output_suffix="_final.md",
            stage_name="Transcription Finalization",
            logger=logger
        )
        if not to_process:
            return

        async def finalize_task(review_file):
            await self._finalize_transcription_task(review_file, transcription_dirs, final_dir)

        await run_tasks_for_files(
            to_process,
            finalize_task,
            logger=logger,
            stage_name="Transcription Finalization"
        )

    async def _finalize_transcription_task(self, review_file, transcription_dirs, final_dir):
        # This is the core LLM call for producing a final transcription for a single page.
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
                str(self.project.get_prompt_path("finalize_transcription")),
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
        # This stage translates the finalized transcriptions using all configured models and runs.
        logger.info(f"Starting translation for project {self.project.name}")
        final_dir = self.project.transcription_final_dir
        if not final_dir.exists():
            logger.error("No finalized transcriptions found. Run finalize_transcriptions first.")
            return

        final_files = sorted(final_dir.glob("*_final.md"))
        page_ids = [f.stem.replace("_final", "") for f in final_files]
        page_id_to_file = {pid: f for pid, f in zip(page_ids, final_files)}

        for config in self.project.translation:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                out_dir = self.project.translation_dir / f"translated_{config.model}{run_suffix}"
                FileManager.ensure_dir(out_dir)

                input_files = [page_id_to_file[pid] for pid in page_ids]
                to_process_files = log_and_filter_unprocessed(
                    input_files=input_files,
                    output_dir=out_dir,
                    output_suffix="_translated.md",
                    stage_name=f"Translation{run_suffix} ({config.model})",
                    logger=logger
                )
                to_process_ids = [f.stem.replace("_final", "") for f in to_process_files]
                if not to_process_ids:
                    continue

                tasks = []
                for idx, page_id in enumerate(page_ids):
                    if page_id in to_process_ids:
                        tasks.append(self._run_translation_task(config, page_id, idx, page_ids, page_id_to_file, out_dir))

                logger.info(f"Starting translation for {len(to_process_ids)} files for {config.model}{run_suffix}...")
                await asyncio.gather(*tasks)
        logger.info("Translation step complete.")

    async def _run_translation_task(self, config, page_id, idx, page_ids, page_id_to_file, out_dir):
        # This is the core LLM call for translating a single finalized transcription.
        final_file = page_id_to_file[page_id]
        final_text = FileManager.read_text(final_file)
        previous_source_text = None
        if idx > 0:
            prev_page_id = page_ids[idx-1]
            prev_file = page_id_to_file[prev_page_id]
            previous_source_text = FileManager.read_text(prev_file)

        prompt = self.llm_service.render_prompt(
            str(self.project.get_prompt_path("translate")),
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
        # Returns all translation run directories for this project. Used for diffing and review.
        dirs = []
        for config in self.project.translation:
            for run in range(config.runs):
                dir_path = self.project.translation_run_dir(config.model, run + 1)
                dirs.append(str(dir_path) if as_str else dir_path)
        return dirs

    def generate_translation_diffs(self) -> None:
        # Compares all translation runs and writes out diffs. This is the basis for translation review and consensus.
        logger.info(f"Generating translation diffs for project {self.project.name}")
        translation_dirs = self._get_translation_dirs(as_str=True)
        labels = []
        for config in self.project.translation:
            for run in range(config.runs):
                dir_path = self.project.translation_run_dir(config.model, run + 1)
                metadata_path = dir_path / "run.yaml"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                        labels.append(metadata["name"])
                else:
                    labels.append(dir_path.name)

        diffs = compare_multiple_folders(
            folders=translation_dirs, labels=labels, file_pattern="*.md"
        )
        diffs_dir = self.project.translation_diffs_dir
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
        # This stage uses LLMs to review the diffs between translation runs, producing rationale for consensus.
        logger.info(f"Starting translation review for project {self.project.name}")
        diffs_dir = self.project.translation_diffs_dir
        if not diffs_dir.exists():
            logger.error("No translation diffs directory found. Run generate_translation_diffs first.")
            return

        reviewed_dir = self.project.translation_reviewed_dir
        FileManager.ensure_dir(reviewed_dir)
        input_files = sorted(diffs_dir.glob("*_diff.md"))
        to_process = log_and_filter_unprocessed(
            input_files=input_files,
            output_dir=reviewed_dir,
            output_suffix="_reviewed.md",
            stage_name="Translation Review",
            logger=logger
        )
        if not to_process:
            return

        async def review_task(diff_file):
            diffs_content = FileManager.read_text(diff_file)
            review_prompt = self.llm_service.render_prompt(
                str(self.project.get_prompt_path("translation_review")),
                {"diff_content": diffs_content}
            )
            for config in self.project.translation:
                await self._review_translation_task(diff_file, config, review_prompt, reviewed_dir)

        await run_tasks_for_files(
            to_process,
            review_task,
            logger=logger,
            stage_name="Translation Review"
        )

    async def _review_translation_task(self, diff_file, config, review_prompt, reviewed_dir):
        # This is the core LLM call for reviewing a single translation diff file. Used by the review_translations stage.
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
        # This stage produces the final, unified translation for each page, using the review rationale and all originals.
        logger.info(f"Starting translation finalization for project {self.project.name}")
        reviewed_dir = self.project.translation_reviewed_dir
        if not reviewed_dir.exists():
            logger.error("No reviewed translation directory found. Run review_translations first.")
            return

        translation_dirs = self._get_translation_dirs(as_str=False)
        final_dir = self.project.translation_final_dir
        FileManager.ensure_dir(final_dir)

        input_files = sorted(reviewed_dir.glob("*_reviewed.md"))
        to_process = log_and_filter_unprocessed(
            input_files=input_files,
            output_dir=final_dir,
            output_suffix="_final.md",
            stage_name="Translation Finalization",
            logger=logger
        )
        if not to_process:
            return

        async def finalize_task(review_file):
            await self._finalize_translation_task(review_file, translation_dirs, final_dir)

        await run_tasks_for_files(
            to_process,
            finalize_task,
            logger=logger,
            stage_name="Translation Finalization"
        )

    async def _finalize_translation_task(self, review_file, translation_dirs, final_dir):
        # This is the core LLM call for producing a final translation for a single page.
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
                final_trans_path = self.project.transcription_final_dir / f"{page_id}_final.md"
                if final_trans_path.exists():
                    final_text = FileManager.read_text(final_trans_path)
                    logger.info(f"Consensus for {page_id}: using finalized transcription as final translation.")
                else:
                    logger.warning(f"Consensus for {page_id}: no translation or finalized transcription found.")
                    return
        else:
            originals_block = "".join([f"## {label}\n{text}\n\n" for label, text in originals.items()])
            prompt = self.llm_service.render_prompt(
                str(self.project.get_prompt_path("finalize_translation")),
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
        # This is the main entry point for running the pipeline. It can run all or selected stages, and supports partial restarts.
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
            self.generate_transcription_diffs()
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
