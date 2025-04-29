import logging
from typing import List, Optional
import asyncio
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from openai import OpenAI
import textwrap

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

    def _log_prefix(self, stage: str, model: Optional[str] = None, run: int | None = None) -> str:
        if model is not None and run is not None:
            return f"[{stage}] [{model}] [run {run}]"
        elif model is not None:
            return f"[{stage}] [{model}]"
        else:
            return f"[{stage}]"

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
                log_prefix = self._log_prefix("Transcription", config.model, run + 1)
                to_process = log_and_filter_unprocessed(
                    input_files=image_files,
                    output_dir=output_dir,
                    output_suffix="_transcribed.md",
                    stage_name=f"{log_prefix}",
                    logger=logger
                )
                if not to_process:
                    continue
                logger.info(f"{log_prefix} Starting transcription stage for {len(to_process)} files.")
                async def process_and_save(img_path):
                    logger.info(f"{log_prefix} Starting file: {img_path.name}")
                    try:
                        transcription = await transcribe_image(
                            llm_service=self.llm_service,
                            image_path=str(img_path),
                            ocr_prompt=ocr_prompt_contents,
                            model=config.model
                        )
                        output_path = output_dir / f"{Path(img_path).stem}_transcribed.md"
                        with open(output_path, 'w') as f:
                            f.write(transcription)
                        preview = textwrap.shorten(transcription.replace('\n', ' '), width=80, placeholder='...')
                        logger.info(f"{log_prefix} Finished file: {img_path.name} -> {output_path} | Preview: {preview}")
                    except Exception as e:
                        logger.error(f"{log_prefix} Error processing {img_path.name}: {e}")
                await run_tasks_for_files(
                    to_process,
                    process_and_save,
                    logger=logger,
                    stage_name=f"{log_prefix}"
                )
                logger.info(f"{log_prefix} Transcription stage complete.")

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
        for config in self.project.transcription:
            for run in range(config.runs):
                dir_path = self.project.transcription_run_dir(config.model, run + 1)
                log_prefix = self._log_prefix("Transcription", config.model, run + 1)
                if not dir_path.exists():
                    logger.warning(f"{log_prefix} Skipping missing transcription run directory: {dir_path}")
        transcription_dirs = []
        labels = []
        for config in self.project.transcription:
            for run in range(config.runs):
                dir_path = self.project.transcription_run_dir(config.model, run + 1)
                log_prefix = self._log_prefix("Transcription", config.model, run + 1)
                if dir_path.exists():
                    transcription_dirs.append(str(dir_path))
                    metadata_path = dir_path / "run.yaml"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = yaml.safe_load(f)
                            labels.append(metadata["name"])
                    else:
                        labels.append(dir_path.name)
                else:
                    logger.warning(f"{log_prefix} Skipping missing transcription run directory: {dir_path}")

        # DEBUG: Print the directories and labels used for diffing
        logger.info(f"[DIFF DEBUG] transcription_dirs: {transcription_dirs}")
        logger.info(f"[DIFF DEBUG] labels: {labels}")

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
        logger.info(self._log_prefix("Transcription Review", None) + f" Starting transcription review for project {self.project.name}")
        if not self.project.transcription_review:
            logger.warning(self._log_prefix("Transcription Review", None) + " No transcription review configuration found")
            return

        diffs_dir = self.project.transcription_diffs_dir
        if not diffs_dir.exists():
            logger.error(self._log_prefix("Transcription Review", None) + " No diffs directory found. Run generate_diffs first.")
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

        log_prefix = "[Transcription Review]"
        async def review_task(diff_file):
            logger.info(f"{log_prefix} Starting file: {diff_file.name}")
            diffs_content = FileManager.read_text(diff_file)
            for config in self.project.transcription_review:
                try:
                    await self._review_transcription_task(diff_file, config, review_prompt, diffs_content, reviewed_dir)
                    reviewed_path = reviewed_dir / diff_file.name.replace("_diff.md", "_transcribed_reviewed.md")
                    if reviewed_path.exists():
                        content = FileManager.read_text(reviewed_path)
                        preview = textwrap.shorten(content.replace('\n', ' '), width=80, placeholder='...')
                        logger.info(f"{log_prefix} Finished file: {diff_file.name} -> {reviewed_path} | Preview: {preview} (model: {config.model})")
                    else:
                        logger.info(f"{log_prefix} Finished file: {diff_file.name} -> {reviewed_path} (model: {config.model}) [file missing]")
                except Exception as e:
                    logger.error(f"{log_prefix} Error processing {diff_file.name} (model: {config.model}): {e}")
        await run_tasks_for_files(
            to_process,
            review_task,
            logger=logger,
            stage_name=log_prefix
        )
        logger.info(f"{log_prefix} Transcription review stage complete.")

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
        logger.info(self._log_prefix("Transcription Finalization", None) + f" Starting transcription finalization for project {self.project.name}")
        reviewed_dir = self.project.transcription_reviewed_dir
        if not reviewed_dir.exists():
            logger.error(self._log_prefix("Transcription Finalization", None) + " No reviewed directory found. Run review_transcriptions first.")
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

        log_prefix = "[Transcription Finalization]"
        async def finalize_task(review_file):
            logger.info(f"{log_prefix} Starting file: {review_file.name}")
            try:
                await self._finalize_transcription_task(review_file, transcription_dirs, final_dir)
                final_path = final_dir / review_file.name.replace("_transcribed_reviewed.md", "_final.md")
                if final_path.exists():
                    content = FileManager.read_text(final_path)
                    preview = textwrap.shorten(content.replace('\n', ' '), width=80, placeholder='...')
                    logger.info(f"{log_prefix} Finished file: {review_file.name} -> {final_path} | Preview: {preview}")
                else:
                    logger.info(f"{log_prefix} Finished file: {review_file.name} -> {final_path} [file missing]")
            except Exception as e:
                logger.error(f"{log_prefix} Error processing {review_file.name}: {e}")
        await run_tasks_for_files(
            to_process,
            finalize_task,
            logger=logger,
            stage_name=log_prefix
        )
        logger.info(f"{log_prefix} Transcription finalization stage complete.")

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
        logger.info(self._log_prefix("Translation", None) + f" Starting translation for project {self.project.name}")
        final_dir = self.project.transcription_final_dir
        if not final_dir.exists():
            logger.error(self._log_prefix("Translation", None) + " No finalized transcriptions found. Run finalize_transcriptions first.")
            return

        final_files = sorted(final_dir.glob("*_final.md"))
        page_ids = [f.stem.replace("_final", "") for f in final_files]
        page_id_to_file = {pid: f for pid, f in zip(page_ids, final_files)}

        for config in self.project.translation:
            for run in range(config.runs):
                run_suffix = f"_{run+1}" if config.runs > 1 else ""
                out_dir = self.project.translation_dir / f"translated_{config.model}{run_suffix}"
                FileManager.ensure_dir(out_dir)
                log_prefix = self._log_prefix("Translation", config.model, run + 1)
                input_files = [page_id_to_file[pid] for pid in page_ids]
                to_process_files = log_and_filter_unprocessed(
                    input_files=input_files,
                    output_dir=out_dir,
                    output_suffix="_translated.md",
                    stage_name=f"{log_prefix}",
                    logger=logger
                )
                to_process_ids = [f.stem.replace("_final", "") for f in to_process_files]
                if not to_process_ids:
                    continue

                logger.info(f"{log_prefix} Starting translation stage for {len(to_process_ids)} files.")
                async def process_and_save(page_id):
                    logger.info(f"{log_prefix} Starting file: {page_id}")
                    try:
                        idx = page_ids.index(page_id)
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
                        translation = await self.llm_service.chat(
                            model=config.model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        assert translation is not None, "Translation is None"
                        out_path = out_dir / f"{page_id}_translated.md"
                        FileManager.write_text(out_path, translation)
                        preview = textwrap.shorten(translation.replace('\n', ' '), width=80, placeholder='...')
                        logger.info(f"{log_prefix} Finished file: {page_id} -> {out_path} | Preview: {preview}")
                    except Exception as e:
                        logger.error(f"{log_prefix} Error processing {page_id}: {e}")
                await run_tasks_for_files(
                    to_process_ids,
                    process_and_save,
                    logger=logger,
                    stage_name=f"{log_prefix}"
                )
                logger.info(f"{log_prefix} Translation stage complete.")
        logger.info(self._log_prefix("Translation", None) + " Translation step complete.")

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
        logger.info(self._log_prefix("Translation", None) + f" Generating translation diffs for project {self.project.name}")
        for config in self.project.translation:
            for run in range(config.runs):
                dir_path = self.project.translation_run_dir(config.model, run + 1)
                log_prefix = self._log_prefix("Translation", config.model, run + 1)
                if not dir_path.exists():
                    logger.warning(f"{log_prefix} Skipping missing translation run directory: {dir_path}")
        translation_dirs = []
        labels = []
        for config in self.project.translation:
            for run in range(config.runs):
                dir_path = self.project.translation_run_dir(config.model, run + 1)
                log_prefix = self._log_prefix("Translation", config.model, run + 1)
                if dir_path.exists():
                    translation_dirs.append(str(dir_path))
                    metadata_path = dir_path / "run.yaml"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = yaml.safe_load(f)
                            labels.append(metadata["name"])
                    else:
                        labels.append(dir_path.name)
                else:
                    logger.warning(f"{log_prefix} Skipping missing translation run directory: {dir_path}")

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
        logger.info(self._log_prefix("Translation Review", None) + f" Starting translation review for project {self.project.name}")
        diffs_dir = self.project.translation_diffs_dir
        if not diffs_dir.exists():
            logger.error(self._log_prefix("Translation Review", None) + " No translation diffs directory found. Run generate_translation_diffs first.")
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

        log_prefix = "[Translation Review]"
        async def review_task(diff_file):
            logger.info(f"{log_prefix} Starting file: {diff_file.name}")
            diffs_content = FileManager.read_text(diff_file)
            review_prompt = self.llm_service.render_prompt(
                str(self.project.get_prompt_path("translation_review")),
                {"diff_content": diffs_content}
            )
            for config in self.project.translation:
                try:
                    await self._review_translation_task(diff_file, config, review_prompt, reviewed_dir)
                    reviewed_path = reviewed_dir / diff_file.name.replace("_diff.md", "_reviewed.md")
                    if reviewed_path.exists():
                        content = FileManager.read_text(reviewed_path)
                        preview = textwrap.shorten(content.replace('\n', ' '), width=80, placeholder='...')
                        logger.info(f"{log_prefix} Finished file: {diff_file.name} -> {reviewed_path} | Preview: {preview} (model: {config.model})")
                    else:
                        logger.info(f"{log_prefix} Finished file: {diff_file.name} -> {reviewed_path} (model: {config.model}) [file missing]")
                except Exception as e:
                    logger.error(f"{log_prefix} Error processing {diff_file.name} (model: {config.model}): {e}")
        await run_tasks_for_files(
            to_process,
            review_task,
            logger=logger,
            stage_name=log_prefix
        )
        logger.info(f"{log_prefix} Translation review stage complete.")

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
        logger.info(self._log_prefix("Translation Finalization", None) + f" Starting translation finalization for project {self.project.name}")
        reviewed_dir = self.project.translation_reviewed_dir
        if not reviewed_dir.exists():
            logger.error(self._log_prefix("Translation Finalization", None) + " No reviewed translation directory found. Run review_translations first.")
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

        log_prefix = "[Translation Finalization]"
        async def finalize_task(review_file):
            logger.info(f"{log_prefix} Starting file: {review_file.name}")
            try:
                await self._finalize_translation_task(review_file, translation_dirs, final_dir)
                final_path = final_dir / review_file.name.replace("_reviewed.md", "_final.md")
                if final_path.exists():
                    content = FileManager.read_text(final_path)
                    preview = textwrap.shorten(content.replace('\n', ' '), width=80, placeholder='...')
                    logger.info(f"{log_prefix} Finished file: {review_file.name} -> {final_path} | Preview: {preview}")
                else:
                    logger.info(f"{log_prefix} Finished file: {review_file.name} -> {final_path} [file missing]")
            except Exception as e:
                logger.error(f"{log_prefix} Error processing {review_file.name}: {e}")
        await run_tasks_for_files(
            to_process,
            finalize_task,
            logger=logger,
            stage_name=log_prefix
        )
        logger.info(f"{log_prefix} Translation finalization stage complete.")

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
        logger.info(self._log_prefix("Pipeline", None) + f" Starting pipeline for project {self.project.name}")
        
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
        logger.info(self._log_prefix("Pipeline", None) + f" Pipeline complete for project {self.project.name}")


def create_project(name: str, input_file: str) -> Project:
    """Create a new project with default configuration"""
    return Project.create(name, input_file)


def load_project(name: str) -> Project:
    """Load an existing project"""
    return Project.from_yaml(name)
