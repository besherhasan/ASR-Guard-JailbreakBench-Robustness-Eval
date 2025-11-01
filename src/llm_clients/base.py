from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class LLMConfig:
    """Configuration tuple describing an LLM under evaluation."""

    name: str
    provider: str
    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[Iterable[str]] = None


class LLMClient(abc.ABC):
    """Abstract interface every model wrapper must implement."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abc.abstractmethod
    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for a single prompt."""

    def shutdown(self) -> None:
        """Optional cleanup hook for network clients."""
        return None


def build_client(config: LLMConfig) -> LLMClient:
    """Factory method to instantiate a concrete LLM client."""
    provider = config.provider.lower()

    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(config)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(config)
    if provider in {"hf", "huggingface"}:
        from .huggingface_client import HuggingFaceClient

        return HuggingFaceClient(config)

    raise ValueError(f"Unsupported provider '{config.provider}' for model '{config.name}'")
