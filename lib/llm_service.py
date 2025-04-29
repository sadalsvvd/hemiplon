import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import jinja2
from typing import Optional, Iterable
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

class LLMService:
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 5):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        assert self.api_key, "OPENAI_API_KEY is not set"
        self.client = OpenAI(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def chat(self, model: str, messages: Iterable[ChatCompletionMessageParam], max_tokens: int = 24000) -> str:
        async with self.semaphore:
            return await asyncio.to_thread(self._blocking_chat, model, messages, max_tokens)

    def _blocking_chat(self, model: str, messages: Iterable[ChatCompletionMessageParam], max_tokens: int) -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
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