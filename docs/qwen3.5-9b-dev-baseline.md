# Qwen3.5-9B development baseline

This is a pre-training development-set check, not a blind benchmark or a claim
about generalizable attack success.

- Model: `Qwen3.5-9B`, aligned local checkpoint on DGX Spark.
- Prompt set: 24 synthetic Chinese development prompts from
  `data/prompts/safety_baseline_dev_zh.txt` (SHA-256
  `05a3b57db0272678204e2b4a6c8145377a2c15f852e4e60eaa9c2d1560a3d1a8`).
- Generation: thinking disabled, greedy decoding, at most 256 new tokens.
  All 24 responses were non-empty and ended with EOS.
- Judge: external DeepSeek `deepseek-chat`, protocol version
  `be1b27c6dd14e8989a8974ea02911e771ef996d62c93c5af0a48f03780a9d414`.
- Result: 24/24 classified as refusal; 0 compliance, 0 invalid, 0 unknown.
  Judge coverage 100%; refusal rate 100%; attack success rate 0%.
- Baseline response JSONL SHA-256:
  `7083a2b136f3445dff5a803fe62a36e712a6e0cc7133e9c85003019ff27f1149`.
- Model `config.json` SHA-256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
- Runtime: PyTorch `2.10.0a0+b558c986e8.nv25.11`, CUDA `13.0`,
  Transformers `5.17.0`.

Raw prompts, responses, Judge labels, and cache remain in DGX `runtime/outputs/`
and are not committed. The 24 labels come from one automated Judge; they have
not been independently adjudicated. The prompt set is for development only and
must not be used as the final held-out evaluation set.
