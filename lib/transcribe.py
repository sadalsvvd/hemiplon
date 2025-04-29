from dotenv import load_dotenv
import os
import base64
import asyncio
import logging
from openai import OpenAI
from pathlib import Path
import time
from typing import List, Dict
import argparse
import difflib
import re
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
assert api_key, "OPENAI_API_KEY is not set"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def write_run_metadata(output_dir: str | Path, run_name: str) -> None:
    """
    Write run metadata to a YAML file in the output directory.
    
    Args:
        output_dir: Output directory path
        run_name: Name of the run
    """
    metadata = {"name": run_name}
    metadata_path = Path(output_dir) / "run.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Run metadata written to {metadata_path}")

# Function to write transcription to a markdown file
def write_transcription(image_name: str, transcription: str, outpath_postfix: str = "", output_dir: str | Path = "output"):
    """
    Write transcription to a markdown file in the specified output directory.
    
    Args:
        image_name: Name of the input image
        transcription: Transcription text to write
        outpath_postfix: Optional postfix for output directory name
        output_dir: Base output directory path
    """
    # Create output directory with postfix if provided
    output_dir = Path(output_dir) / f"transcribed{outpath_postfix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write run metadata if it doesn't exist
    if not (output_dir / "run.yaml").exists():
        run_name = f"transcribed{outpath_postfix}"
        write_run_metadata(output_dir, run_name)
    
    # Write to output directory
    output_path = output_dir / f"{Path(image_name).stem}_transcribed.md"
    with open(output_path, 'w') as file:
        file.write(transcription)
    logger.info(f"Transcription written to {output_path}")

# Function to process a single image
async def process_image(
    client: OpenAI,
    image_path: str,
    ocr_prompt: str,
    semaphore: asyncio.Semaphore,
    outpath_postfix: str = "",
    model: str = "gpt-4.1",
    output_dir: str | Path = "output"
):
    async with semaphore:  # This ensures we only have max_concurrent requests at once
        start_time = time.time()
        image_name = Path(image_path).name
        logger.info(f"Starting processing for image: {image_name}")
        
        try:
            base64_image = encode_image(image_path)
            logger.info(f"Image {image_name} encoded to base64. Length: {len(base64_image)}")
            
            ocr_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": ocr_prompt },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=24000,
            )
            
            transcription = ocr_completion.choices[0].message.content
            end_time = time.time()
            logger.info(f"Completed processing for {image_name} in {end_time - start_time:.2f} seconds")

            assert transcription is not None, f"Transcription is None for {image_name}"
            
            # Write transcription immediately
            write_transcription(image_name, transcription, outpath_postfix, output_dir)
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing {image_name}: {str(e)}")
            raise

# Main function to process a directory
async def process_directory(
    directory_path: str,
    ocr_prompt_path: str,
    max_concurrent: int = 3,
    start_index: int = 0,
    end_index: int | None = None,
    outpath_postfix: str = "",
    model: str = "gpt-4.1",
    output_dir: str | Path = "output"
):
    client = OpenAI(api_key=api_key)
    
    # Read OCR prompt
    with open(ocr_prompt_path, "r") as file:
        ocr_prompt_contents = file.read()
    logger.info(f"OCR prompt loaded from {ocr_prompt_path}")
    
    # Get all image files in the directory and sort them
    image_files = sorted([f for f in Path(directory_path).glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    logger.info(f"Found {len(image_files)} image files in {directory_path}")
    
    # Apply start and end index bounds
    if end_index is None:
        end_index = len(image_files)
    image_files = image_files[start_index:end_index]
    logger.info(f"Processing images from index {start_index} to {end_index}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all images
    tasks = [
        process_image(client, str(img), ocr_prompt_contents, semaphore, outpath_postfix, model, output_dir)
        for img in image_files
    ]
    
    # Process all images concurrently, with max_concurrent limit
    await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process images and generate transcriptions')
    parser.add_argument('--directory', type=str, default="images/pages", help='Directory containing images')
    parser.add_argument('--prompt', type=str, default="prompts/transcribe.md", help='Path to OCR prompt')
    parser.add_argument('--start', type=int, default=0, help='Start processing from this index')
    parser.add_argument('--end', type=int, default=10, help='Process up to this index (exclusive)')
    parser.add_argument('--concurrent', type=int, default=3, help='Maximum number of concurrent requests')
    parser.add_argument('--postfix', type=str, default="_gpt-4.1_high-detail", help='Optional postfix for output directory')
    parser.add_argument('--model', type=str, default="gpt-4.1", help='OpenAI model to use for transcription')
    
    args = parser.parse_args()
    
    asyncio.run(process_directory(
        args.directory,
        args.prompt,
        max_concurrent=args.concurrent,
        start_index=args.start,
        end_index=args.end,
        outpath_postfix=args.postfix,
        model=args.model
    ))
