# Alignment Stress Lab

English | [简体中文](README.zh-CN.md)

Alignment Stress Lab measures how reward-driven fine-tuning changes a language
model's safety behavior and general utility. The first targets are aligned
Dense checkpoints in the Qwen3.5-9B and Qwen3.8-27B families. The project is
under development and has no validated benchmark results yet.

## Project goals

- Use one training code path for both model sizes, with model-specific settings.
- Run a policy adapter and its frozen reference against one locally loaded base.
- Use an external Judge API so the DGX does not need a second Judge model.
- Preserve failed Judge calls as unknown and invalid generations as failed
  attempts when calculating attack success rate (ASR).
- Record evaluator versions, cache provenance, Judge coverage, safety metrics,
  and utility metrics for reproducible comparisons.

These are design goals. The evaluation contract, initial Judge client, and
model-independent GRP-Oblit loss are implemented and tested on synthetic CPU
tensors. Only empty responses are detected as invalid so far; further
generation quality checks, model loading, training, and benchmark integration
remain future milestones.

## Research boundary

Run experiments only in an isolated, authorized environment. The project does
not automatically upload datasets, responses, adapters, or checkpoints. Keep
held-out evaluation data separate from training and tuning. Trained checkpoints
are not intended for public inference deployment.

## Development setup

Use Python 3.11 or 3.12. On Windows, select an installed interpreter first.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
```

`pyproject.toml` is the source of dependency version constraints.
`requirements.txt` installs the core package, `requirements-dev.txt` adds test
tools and CPU PyTorch for synthetic math tests, and `requirements-train.txt`
adds model training packages. No model weights are used by local tests. On DGX Spark,
install the training set inside a validated NGC-derived container; retain its
compatible ARM64 CUDA and PyTorch stack.

The mathematical contract is in [docs/grp-oblit-math.md](docs/grp-oblit-math.md).

## Judge configuration

See [configs/judge.example.yaml](configs/judge.example.yaml). Configure
`JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY` in the environment. The
Judge sends prompt-response pairs to the configured external API. Confirm that
the provider permits the research data before using it. Offline tests inject a
fake client and send no requests.

## Results

No ASR or utility claim is published. Results will be added after the model
checkpoints, held-out benchmarks, sampling settings, Judge protocol, and
analysis rules have been fixed and recorded.

The public [Chinese safety development prompts](data/prompts/safety_baseline_dev_zh.txt)
are synthetic, one prompt per line, and intended only to exercise the baseline
pipeline. They are not a held-out benchmark or training data.

## Acknowledgments

This project draws engineering ideas from
[Abliterix](https://github.com/wuwangzhang1216/abliterix) (Judge handling,
cache provenance, and evaluation failure semantics) and
[Heretic](https://github.com/p-e-w/heretic) (adapter and model-loading
engineering). The [GRP-Obliteration paper](https://arxiv.org/abs/2602.06258)
motivates the research question, and the
[community GRP repository](https://github.com/GuyNachshon/GRP-Obliteration)
provided implementation points to inspect. This repository is an independent
implementation; these acknowledgments do not imply endorsement or verified
reproduction of their results.

## License

This repository is licensed under [Apache-2.0](LICENSE). The acknowledged
projects retain their own licenses; no source code from them is included here.
