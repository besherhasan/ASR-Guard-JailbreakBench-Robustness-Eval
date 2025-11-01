from __future__ import annotations

from typing import Optional

from anthropic import Anthropic  # type: ignore

from .base import LLMClient, LLMConfig


class AnthropicClient(LLMClient):
    """Wrapper around the Anthropic Messages API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.params.get("api_key")
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 512,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stop_sequences": self.config.stop_sequences,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        system = system_prompt or self.config.params.get("default_system_prompt")

        response = self._client.messages.create(  # type: ignore[attr-defined]
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
