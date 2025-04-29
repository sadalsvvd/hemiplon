from pathlib import Path
from typing import List
import logging
import asyncio

def get_unprocessed_files(
    input_dir: Path,
    output_dir: Path,
    input_glob: str,
    output_suffix: str
) -> List[Path]:
    """
    Returns a list of input files in input_dir matching input_glob that do not have a corresponding output file
    with the given output_suffix in output_dir.
    """
    input_files = sorted(input_dir.glob(input_glob))
    # Map input file stems to input files
    input_stem_to_file = {f.stem: f for f in input_files}
    # Find all output files with the given suffix
    output_files = list(output_dir.glob(f"*{output_suffix}"))
    # Extract the stem (without suffix) for each output file
    output_stems = {f.stem.replace(output_suffix.replace('.md','').replace('_',''), '') for f in output_files}
    # Only process input files whose stem is not in output_stems
    to_process = [f for stem, f in input_stem_to_file.items() if stem not in output_stems]
    return to_process


def log_and_filter_unprocessed(
    input_files: List[Path],
    output_dir: Path,
    output_suffix: str,
    stage_name: str,
    logger: logging.Logger
) -> List[Path]:
    output_files = set(f.stem for f in output_dir.glob(f"*{output_suffix}"))
    to_process = [f for f in input_files if f.stem not in output_files]
    skipped = len(input_files) - len(to_process)
    if skipped == 0:
        logger.info(f"{stage_name}: No previous outputs found, starting fresh.")
    else:
        logger.info(f"{stage_name}: Resuming, {skipped} files already processed, {len(to_process)} left.")
    if not to_process:
        logger.info(f"{stage_name}: All files already processed. Nothing to do.")
    return to_process 

async def run_tasks_for_files(
    files,
    process_fn,
    *args,
    logger=None,
    stage_name=None,
    **kwargs
):
    if logger and stage_name:
        logger.info(f"{stage_name}: Starting processing for {len(files)} files...")
    tasks = [process_fn(file, *args, **kwargs) for file in files]
    await asyncio.gather(*tasks)
    if logger and stage_name:
        logger.info(f"{stage_name}: Step complete.") 