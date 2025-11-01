from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from datasets import DatasetDict, load_dataset

from .llm_clients.base import LLMClient


class HarnessResult(Protocol):
    """Protocol describing the minimal information we need from the harness."""

    @property
    def behavior_id(self) -> str: ...

    @property
    def behavior_category(self) -> str: ...

    @property
    def attack_success(self) -> bool: ...


@dataclass
class ModelEvaluationSummary:
    model_name: str
    asr_overall: float
    asr_by_category: Dict[str, float]
    total_attacks: int


class JailbreakBenchRunner:
    """Thin wrapper around the official JailbreakBench harness."""

    def __init__(
        self,
        dataset_name: str = "JailbreakBench/JBB-Behaviors",
        split: str = "test",
        max_examples: Optional[int] = None,
        harness_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_examples = max_examples
        self.dataset = load_dataset(dataset_name, split=split)
        self._harness_module = self._load_harness_module()
        raw_kwargs = harness_kwargs or {}
        self.harness_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    @staticmethod
    def _load_harness_module():
        try:
            return importlib.import_module("jailbreakbench.harness")
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The official JailbreakBench harness is required. "
                "Install it with `pip install jailbreakbench`."
            ) from exc

    def _build_harness(self, model_client: LLMClient):
        if hasattr(self._harness_module, "JailbreakBenchHarness"):
            harness_cls = getattr(self._harness_module, "JailbreakBenchHarness")
            return harness_cls(model_client.generate, **self.harness_kwargs)
        if hasattr(self._harness_module, "create_harness"):
            return self._harness_module.create_harness(model_client.generate, **self.harness_kwargs)
        raise AttributeError(
            "Unsupported JailbreakBench harness API. Please update src/harness_runner.py "
            "to match the installed harness version."
        )

    def evaluate(self, model_client: LLMClient) -> ModelEvaluationSummary:
        harness = self._build_harness(model_client)

        records = None
        if hasattr(harness, "run"):
            try:
                records = harness.run(
                    dataset=self.dataset,
                    limit=self.max_examples,
                )
            except TypeError:
                records = harness.run(
                    dataset_name=self.dataset_name,
                    split=self.split,
                    limit=self.max_examples,
                )
        elif hasattr(harness, "evaluate_model"):
            try:
                records = harness.evaluate_model(
                    dataset=self.dataset,
                    limit=self.max_examples,
                )
            except TypeError:
                records = harness.evaluate_model(
                    dataset_name=self.dataset_name,
                    split=self.split,
                    limit=self.max_examples,
                )
        if records is None:
            raise AttributeError(
                "Unsupported harness interface. Expected `.run` or `.evaluate_model` methods."
            )

        attack_records: List[HarnessResult] = list(records)
        if self.max_examples:
            attack_records = attack_records[: self.max_examples]

        totals = defaultdict(int)
        successes = defaultdict(int)

        for record in attack_records:
            category = getattr(record, "behavior_category", "unknown")
            totals[category] += 1
            if getattr(record, "attack_success", False):
                successes[category] += 1

        asr_by_category = {
            category: successes[category] / totals[category]
            for category in totals
            if totals[category] > 0
        }

        overall_asr = sum(successes.values()) / max(1, sum(totals.values()))

        return ModelEvaluationSummary(
            model_name=model_client.config.name,
            asr_overall=overall_asr,
            asr_by_category=asr_by_category,
            total_attacks=sum(totals.values()),
        )
