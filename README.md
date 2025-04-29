# hemiplon

Dependencies:

- `brew install pkg-config poppler`
  - https://github.com/jalan/pdftotext/issues/9

# Project Structure

`hemiplon` uses a folder-based project system to manage transcription, translation, and review run data. Each project is organized as follows:

```
projects/
  <project_name>/
    config.yaml         # Project configuration
    images/            # Extracted and processed page images
      page_001.jpg
      page_002.jpg
      ...
    transcription/
      transcribed_<model>_<run>/  # Each transcription run
        run.yaml                  # Run metadata
        page_001_transcribed.md
        ...
      diffs/                     # Per-page diffs between runs
        page_001_diff.md
        ...
      transcribed_reviewed/      # Per-page review rationale
        page_001_transcribed_reviewed.md
        ...
      final/                     # Final unified transcription per page
        page_001_final.md
        ...
    translation/
      translated_<model>_<run>/  # Each translation run
        run.yaml
        page_001_translated.md
        ...
      diffs/                     # Per-page diffs between translation runs
        page_001_diff.md
        ...
      reviewed/                  # Per-page translation review rationale
        page_001_reviewed.md
        ...
      final/                     # Final unified translation per page
        page_001_final.md
        ...
```

Each run directory contains a `run.yaml` file that identifies the run by name, making it easier to track and compare different runs without relying on directory names.

# CLI Usage

## Creating a Project

Create a new project with default settings:
```bash
uv run python -m cli create <name> <input_filename>
```

This will:
1. Create a project directory in `projects/<name>`
2. Generate a default `config.yaml` with transcription and translation settings
3. Set up the necessary subdirectories

## Running the Pipeline

The pipeline consists of multiple stages, which can be run independently or together:

### Transcription Pipeline Stages
- `pdf`: Convert PDF to individual page images
- `transcription`: Run OCR/transcription on images
- `transcription-diff`: Generate per-page diffs between different transcription runs
- `transcription-review`: Review diffs and generate rationale for each page
- `transcription-finalize`: Produce a final unified transcription per page

### Translation Pipeline Stages
- `translation`: Translate each reviewed transcription into English (or other target language)
- `translation-diff`: Generate per-page diffs between different translation runs
- `translation-review`: Review translation diffs and generate rationale for each page
- `translation-finalize`: Produce a final unified translation per page

### Running the Complete Pipeline

Run all stages (transcription and translation):
```bash
uv run python -m cli run <name>
```

Run specific stages:
```bash
uv run python -m cli run <name> --stages pdf transcription transcription-diff transcription-review transcription-finalize translation translation-diff translation-review translation-finalize
```

Or just the translation pipeline (requires transcription to be present):
```bash
uv run python -m cli run <name> --stages translation translation-diff translation-review translation-finalize
```

### Transcription Options

When running transcription, you can process specific page ranges:

```bash
# Process first 5 pages (0-4)
uv run python -m cli run <name> --stages transcription --end 5

# Process pages 10-20
uv run python -m cli run <name> --stages transcription --start 10 --end 20

# Process from page 15 onwards
uv run python -m cli run <name> --stages transcription --start 15
```

Note: Page indices are zero-based:
- `--start`: First page to process (inclusive)
- `--end`: Last page to process (exclusive)

## Project Configuration

Each project's `config.yaml` defines:
- Input file path
- Transcription settings per model:
  - Number of runs
  - Maximum concurrent requests
  - OCR prompt path
- Translation settings per model:
  - Number of runs
  - Model name

## Pipeline Details

- **Transcription pipeline**: Converts PDF to images, runs OCR, generates diffs, reviews, and finalizes a unified transcription per page.
- **Translation pipeline**: Translates each reviewed transcription, generates diffs between translation runs, reviews, and finalizes a unified translation per page.
- **Consensus handling**: If all models agree, the pipeline skips review/finalization and uses the consensus output directly.
- **All outputs are organized by stage and model/run for full traceability.**

## Next Steps

- Integrate cleaning functionality
- Add support for additional languages or translation targets
- Further improve review/finalization prompts and logic

