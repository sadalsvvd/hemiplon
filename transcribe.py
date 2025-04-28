from dotenv import load_dotenv
import os
import base64
import asyncio
import logging
from openai import OpenAI
from pathlib import Path
import time
from typing import List

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

# Function to write transcription to a markdown file
def write_transcription(image_name: str, transcription: str):
    # Create output directory if it doesn't exist
    output_dir = Path("output/transcribed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    semaphore: asyncio.Semaphore
):
    async with semaphore:  # This ensures we only have max_concurrent requests at once
        start_time = time.time()
        image_name = Path(image_path).name
        logger.info(f"Starting processing for image: {image_name}")
        
        try:
            base64_image = encode_image(image_path)
            logger.info(f"Image {image_name} encoded to base64. Length: {len(base64_image)}")
            
            ocr_completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": ocr_prompt },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )
            
            transcription = ocr_completion.choices[0].message.content
            end_time = time.time()
            logger.info(f"Completed processing for {image_name} in {end_time - start_time:.2f} seconds")

            assert transcription is not None, f"Transcription is None for {image_name}"
            
            # Write transcription immediately
            write_transcription(image_name, transcription)
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
    end_index: int | None = None
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
        process_image(client, str(img), ocr_prompt_contents, semaphore)
        for img in image_files
    ]
    
    # Process all images concurrently, with max_concurrent limit
    await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    directory_path = "images"  # Directory containing images
    ocr_prompt_path = "prompts/transcribe.md"  # Path to OCR prompt
    start_index = 0  # Start processing from this index
    end_index = 10  # Process up to this index (exclusive)
    max_concurrent = 3  # Maximum number of concurrent requests
    
    asyncio.run(process_directory(
        directory_path,
        ocr_prompt_path,
        max_concurrent=max_concurrent,
        start_index=start_index,
        end_index=end_index
    ))
# # Translation step using text model
# translation_prompt = f"Translate the following Latin text to English:\n\n{ocr_text}"
# translation_completion = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 { "type": "text", "text": translation_prompt }
#             ],
#         }
#     ],
#     max_tokens=1024,
# )

# translated_text = translation_completion.choices[0].message.content
