import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import jinja2
from typing import Optional, Iterable
from openai.types.chat import ChatCompletionMessageParam
import json
from datetime import datetime
import uuid
from lib.logger import logger
import threading

load_dotenv()

class LLMService:
    # Class-level counters and lock for tracking requests/responses
    total_requests_sent = 0
    total_responses_received = 0
    _counter_lock = threading.Lock()

    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 5):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        assert self.api_key, "OPENAI_API_KEY is not set"
        self.client = OpenAI(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def chat(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        max_tokens: int = 24000,
        request_name: Optional[str] = None
    ) -> str:
        # Prepare file for logging
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        rand_id = uuid.uuid4().hex[:5]
        if request_name:
            log_filename = f"{timestamp}_{request_name}_{rand_id}.txt"
        else:
            log_filename = f"{timestamp}_{rand_id}.txt"

        logger.info(f"[LLMService] Calling model={model}, max_tokens={max_tokens}, id={rand_id}")

        # Increment and log total requests sent
        with self.__class__._counter_lock:
            self.__class__.total_requests_sent += 1
            logger.info(
                f"[LLMService] Total requests sent: {self.__class__.total_requests_sent} "
                f"(model={model}, id={rand_id})"
            )

        # Write prompt/messages to file immediately
        # Make sure logs directory exists
        os.makedirs("logs", exist_ok=True)
        log_output_path = "logs/" + log_filename

        with open(log_output_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Max tokens: {max_tokens}\n")
            f.write("Prompt/messages:\n")
            f.write(json.dumps(list(messages), indent=2, ensure_ascii=False))
            f.write("\n")

        logger.info(f"[LLMService] Waiting for semaphore (id={rand_id})")
        async with self.semaphore:
            logger.info(f"[LLMService] Acquired semaphore (id={rand_id})")
            try:
                # Actually call the LLM
                logger.info(f"[LLMService] Entering LLM call (id={rand_id})")
                try:
                    # Add a timeout to prevent infinite blocking
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self._blocking_chat, model, messages, max_tokens),
                        timeout=180  # 3 minutes timeout (longer than the 120s in _blocking_chat)
                    )
                    logger.info(f"[LLMService] Exited LLM call (id={rand_id})")
                except asyncio.TimeoutError:
                    logger.error(f"[LLMService] Timeout waiting for LLM call (id={rand_id})")
                    raise TimeoutError(f"LLM call timed out after 180 seconds (id={rand_id})")

                # Increment and log total responses received
                with self.__class__._counter_lock:
                    self.__class__.total_responses_received += 1
                    logger.info(
                        f"[LLMService] Total responses received: {self.__class__.total_responses_received} "
                        f"(model={model}, id={rand_id})"
                    )

                logger.info(f"[LLMService] {timestamp} {rand_id} Response: {result}")
            except Exception as e:
                logger.error(f"[LLMService] Exception in LLM call (id={rand_id}): {e}")
                raise
            finally:
                logger.info(f"[LLMService] Releasing semaphore (id={rand_id})")

        # Write the response to the file
        with open(log_output_path, "a", encoding="utf-8") as f:
            f.write("\n-------\n")
            f.write("Response:\n")
            f.write(result)
            f.write("\n")

        # print(f"[LLMService] Response: {result}")

        return result

    def _blocking_chat(self, model: str, messages: Iterable[ChatCompletionMessageParam], max_tokens: int) -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=120,
        )
        content = completion.choices[0].message.content
        return content if content is not None else ""

    @staticmethod
    def render_prompt(template_path: str, context: dict) -> str:
        template_dir = Path(template_path).parent
        template_name = Path(template_path).name
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template(template_name)
        return template.render(**context) 
