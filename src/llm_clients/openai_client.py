from __future__ import annotations

from typing import Optional

from openai import OpenAI, AsyncOpenAI  # type: ignore

from .base import LLMClient, LLMConfig


class OpenAIClient(LLMClient):
    """Wrapper around the OpenAI responses API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        client_cls = AsyncOpenAI if config.params.get("async_client") else OpenAI
        api_key = config.params.get("api_key")
        self._client = client_cls(api_key=api_key) if api_key else client_cls()

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_output_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = self._client.responses.create(input=messages, **kwargs)  # type: ignore[arg-type]
        text_chunks = [out_item.content[0].text for out_item in response.output]  # type: ignore[index]
        return "".join(text_chunks).strip()
