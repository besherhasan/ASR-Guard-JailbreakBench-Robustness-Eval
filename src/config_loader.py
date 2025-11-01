from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .llm_clients import LLMConfig


def load_llm_configs(config_path: Path) -> List[LLMConfig]:
    """Parse a YAML file containing a list of LLM configurations."""
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    configs: List[LLMConfig] = []
    for entry in payload:
        if "name" not in entry or "model" not in entry or "provider" not in entry:
            raise ValueError(f"LLM config missing required keys: {entry}")
        configs.append(
            LLMConfig(
                name=entry["name"],
                provider=entry["provider"],
                model=entry["model"],
                params=entry.get("params", {}),
                streaming=entry.get("streaming", False),
                max_tokens=entry.get("max_tokens"),
                temperature=entry.get("temperature"),
                top_p=entry.get("top_p"),
                stop_sequences=entry.get("stop_sequences"),
            )
        )
    return configs
