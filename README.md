# ASR-Guard: JailbreakBench Robustness Evaluation

This project evaluates large language model (LLM) robustness against jailbreak attacks using the official [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) harness. It computes attack-success rate (ASR) overall and by behavior category for multiple LLM configurations, producing reproducible JSON/CSV reports suitable for benchmarking or regression tracking.

## Repository Layout

- `configs/models.yaml` – declarative list of 6 LLM configurations (OpenAI, Anthropic, and open-source Hugging Face models).
- `requirements.txt` – runtime dependencies (transformers, datasets, jailbreakbench harness, etc.).
- `src/` – evaluation code:
  - `llm_clients/` – adapters for OpenAI, Anthropic, and local Hugging Face models.
  - `config_loader.py` – YAML parser for LLM configs.
  - `harness_runner.py` – wrapper that drives the official JailbreakBench harness and aggregates per-category ASR.
  - `run_evaluations.py` – CLI entry point to evaluate 4–8 LLMs end-to-end.
- `reports/` – output directory for generated metrics (`.gitkeep` placeholder included).

## Prerequisites

- Python 3.10+
- GPU with at least 24 GB VRAM recommended for local 7B/8B Hugging Face models (adjust configs if using quantized variants).
- Valid API keys exported as environment variables when evaluating OpenAI (`OPENAI_API_KEY`) and Anthropic (`ANTHROPIC_API_KEY`) models.
- Network access to pull the `JailbreakBench/JBB-Behaviors` dataset from Hugging Face Hub and to reach the selected LLM providers.

Install dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

The `jailbreakbench` package supplies the official harness used here; ensure that installation succeeds as it pins judge prompts, scoring, and attack templates.

## Configuring Models

The default `configs/models.yaml` lists six distinct LLMs spanning hosted APIs and local models:

1. `gpt-4o` (OpenAI)
2. `gpt-4o-mini` (OpenAI)
3. `claude-3-5-sonnet-20240620` (Anthropic)
4. `claude-3-haiku-20240307` (Anthropic)
5. `meta-llama/Meta-Llama-3-8B-Instruct` (Hugging Face local inference)
6. `mistralai/Mistral-7B-Instruct-v0.2` (Hugging Face local inference)

You can add or remove entries to target 4–8 models; the CLI consumes whichever list is present. Each entry supports provider-specific parameters (e.g., device map, default safety system prompts, sampling settings). For Hugging Face models, adjust `model_kwargs` to load quantized checkpoints (such as `load_in_8bit: true`) if resources are limited.

## Running an Evaluation

> **Tip:** Run the CLI as a module so relative imports resolve correctly.

```bash
python src/run_evaluations.py \
  --config configs/models.yaml \
  --split test \
  --judge-model gpt-4o-mini \
  --attack-template default \
  --max-examples 200 \
  --output-dir reports
```

Parameters:

- `--config`: path to the LLM configuration YAML (default `configs/models.yaml`).
- `--split`: dataset split to evaluate (`train`, `validation`, or `test` depending on Hugging Face dataset availability).
- `--judge-model`: JailbreakBench judge to score outputs (defaults to `gpt-4o-mini`; align with harness presets you are authorized to run).
- `--attack-template`: jailbreak strategy preset compatible with the harness build you install.
- `--max-examples`: optional limit for debugging; omit to evaluate the full split.
- `--output-dir`: directory for the generated JSON/CSV reports.

During execution:

1. The CLI loads `JailbreakBench/JBB-Behaviors` via `datasets.load_dataset`.
2. For each LLM, the official harness orchestrates jailbreak prompts, collects responses, and uses standardized judges/metrics.
3. Attack records are aggregated to compute ASR overall and per behavior category.
4. Two artifacts are written:
   - `reports/asr_summaries.json`: structured summaries (overall ASR, totals, and per-category breakdowns).
   - `reports/asr_breakdown.csv`: tabular view with ASR columns for every category, ready for plotting or regression dashboards.

No evaluation is executed automatically; the repository ships runnable code only.

## Extending the Pipeline

- **New models:** Append entries to `configs/models.yaml`. Implement additional adapters under `src/llm_clients/` if your provider is not currently supported.
- **Custom harness settings:** Update the CLI options or pass arbitrary keyword arguments through `--judge-model` / `--attack-template`. If the harness version you install expects different parameter names, adjust `src/harness_runner.py`.
- **Experiment tracking:** The JSON report structure is intentionally simple so it can be ingested into tools like MLflow or Weights & Biases.

## Reference Results

The JailbreakBench maintainers collect community submissions that include end-to-end harness outputs. The table below summarizes overall and per-category attack-success rates (ASR) extracted from the AIR attack submission targeting nine models (see [JailbreakBench/jailbreakbench#35](https://github.com/JailbreakBench/jailbreakbench/issues/35), Nov 17 2024). Each column reports the fraction of behaviors (out of 10 per category) that the submitted jailbreak successfully elicited.

| Model | Overall ASR | Disinformation | Economic harm | Expert advice | Fraud/Deception | Government decisions | Harassment/Discrimination | Malware/Hacking | Physical harm | Privacy | Sexual/Adult |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-3-5-sonnet-20240620 | 94% | 100% | 100% | 90% | 100% | 100% | 100% | 100% | 90% | 100% | 60% |
| gpt-4o-2024-05-13 | 95% | 80% | 90% | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 90% |
| gpt-4o-mini-2024-07-18 | 92% | 70% | 80% | 80% | 100% | 100% | 90% | 100% | 100% | 100% | 100% |
| llama-3-70b-chat-hf | 88% | 100% | 80% | 90% | 100% | 100% | 100% | 100% | 70% | 90% | 50% |
| llama-3-8b-chat-hf | 83% | 90% | 100% | 80% | 100% | 100% | 90% | 100% | 50% | 90% | 30% |
| qwen2-0.5b-instruct | 84% | 40% | 90% | 60% | 90% | 90% | 90% | 90% | 100% | 90% | 100% |
| qwen2-1.5b-instruct | 94% | 80% | 90% | 90% | 100% | 90% | 100% | 100% | 100% | 100% | 90% |
| qwen2-72b-instruct | 90% | 70% | 100% | 60% | 100% | 100% | 100% | 100% | 90% | 100% | 80% |
| qwen2-7b-instruct | 92% | 70% | 100% | 80% | 100% | 90% | 90% | 100% | 90% | 100% | 100% |

These numbers illustrate how widely the ASR varies by threat category even when overall robustness appears similar. When comparing to your own runs, ensure that judge, attack template, and harness version match the cited submission.

## Troubleshooting

- If you see `ImportError: jailbreakbench`, ensure `pip install jailbreakbench` succeeded and that the package version exposes either `JailbreakBenchHarness` or `create_harness`. Update `src/harness_runner.py` if the upstream API changes.
- Hugging Face models may require `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121` (or similar) for GPU acceleration.
- When working within a restricted environment, pre-download the dataset (`datasets-cli download JailbreakBench/JBB-Behaviors`) so the harness can function offline.

## License & Attribution

- Dataset: [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (SemEval-2025 Task contributors).
- Code: released for educational evaluation; adapt to your organization’s policy requirements before production use.
