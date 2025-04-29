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
    output/
      transcribed_<model>_<run>/  # Each transcription run
        run.yaml                  # Run metadata
        page_001_transcribed.md
        page_002_transcribed.md
        ...
      transcription_diffs.txt     # Multi-way diffs between runs
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
2. Generate a default `config.yaml` with transcription settings
3. Set up the necessary subdirectories

## Running the Pipeline

The pipeline consists of three stages:
- `pdf`: Convert PDF to individual page images
- `transcription`: Run OCR/transcription on images
- `diffs`: Generate diffs between different transcription runs

Run the complete pipeline:
```bash
uv run python -m cli run <name>
```

Run specific stages:
```bash
uv run python -m cli run <name> --stages pdf transcription
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
- Translation settings

## Next Steps

- Integrate cleaning functionality
- Integrate actual translation, including previous values

