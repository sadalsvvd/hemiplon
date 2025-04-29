from dotenv import load_dotenv
import os
import base64
import logging
from lib.llm_service import LLMService
from openai.types.chat import ChatCompletionMessageParam

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

async def transcribe_image(
    llm_service: LLMService,
    image_path: str,
    ocr_prompt: str,
    model: str = "gpt-4.1"
) -> str:
    """
    Transcribes a single image using the provided LLMService and prompt.
    Returns the transcription string.
    """
    base64_image = encode_image(image_path)
    messages: list[ChatCompletionMessageParam] = [
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
    ]
    transcription = await llm_service.chat(
        model=model,
        messages=messages,
        max_tokens=24000,
    )
    assert transcription is not None, f"Transcription is None for {image_path}"
    return transcription
