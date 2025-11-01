from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    from pathlib import Path as _Path  # type: ignore

    sys.path.append(str(_Path(__file__).resolve().parent))
    from config_loader import load_llm_configs  # type: ignore
    from harness_runner import JailbreakBenchRunner  # type: ignore
    from llm_clients.base import build_client  # type: ignore
else:
    from .config_loader import load_llm_configs
    from .harness_runner import JailbreakBenchRunner
    from .llm_clients.base import build_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple LLMs on JailbreakBench.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models.yaml"),
        help="Path to the YAML file with LLM configurations.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (train/dev/test).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit on number of behaviors to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where JSON/CSV reports will be stored.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model identifier for the JailbreakBench harness.",
    )
    parser.add_argument(
        "--attack-template",
        type=str,
        default="default",
        help="Attack template or jailbreak strategy preset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_llm_configs(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    harness_runner = JailbreakBenchRunner(
        split=args.split,
        max_examples=args.max_examples,
        harness_kwargs={
            "judge_model": args.judge_model,
            "attack_template": args.attack_template,
        },
    )

    summaries = []
    category_matrix: Dict[str, Dict[str, float]] = {}

    for config in tqdm(configs, desc="Models"):
        client = build_client(config)
        try:
            summary = harness_runner.evaluate(client)
            summaries.append(summary)
            category_matrix[summary.model_name] = summary.asr_by_category
        finally:
            client.shutdown()

    # Persist raw summaries
    json_path = args.output_dir / "asr_summaries.json"
    payload = [
        {
            "model_name": summary.model_name,
            "asr_overall": summary.asr_overall,
            "total_attacks": summary.total_attacks,
            "asr_by_category": summary.asr_by_category,
        }
        for summary in summaries
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write tabular overview
    csv_rows = []
    for summary in summaries:
        row = {"model_name": summary.model_name, "asr_overall": summary.asr_overall}
        row.update(summary.asr_by_category)
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows).set_index("model_name")
    df.sort_values("asr_overall", inplace=True)
    df.to_csv(args.output_dir / "asr_breakdown.csv")

    print(f"Saved evaluation summaries to {json_path} and ASR breakdown to asr_breakdown.csv")


if __name__ == "__main__":
    main()
