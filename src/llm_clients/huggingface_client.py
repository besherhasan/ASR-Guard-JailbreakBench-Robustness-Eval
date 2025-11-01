from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import LLMClient, LLMConfig


class HuggingFaceClient(LLMClient):
    """Simple wrapper around a local HF transformers pipeline."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        model_kwargs = config.params.get("model_kwargs", {})
        torch_dtype = model_kwargs.pop("torch_dtype", None)
        if isinstance(torch_dtype, str):
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        self._pipeline = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                config.model,
                device_map=config.params.get("device_map", "auto"),
                **model_kwargs,
            ),
            tokenizer=self._tokenizer,
            max_new_tokens=config.max_tokens or 512,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature is not None and config.temperature > 0,
        )

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        result = self._pipeline(full_prompt, return_full_text=False)
        return result[0]["generated_text"].strip()
