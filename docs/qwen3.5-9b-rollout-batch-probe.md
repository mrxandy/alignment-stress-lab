# Qwen3.5-9B generation batch probe

On 2026-09-24, the DGX Spark container ran the fictional-news proxy generation
probe from commit `a0ac214` on one NVIDIA GB10. This was inference only: no
Judge call, gradient update, or checkpoint write occurred.

- Model `config.json` SHA-256: `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
- 9B profile SHA-256: `8b5f028656cd917b3031a78080e62ab8473dca0a0ff0d24fef32776ae2a0bec3`.
- Fictional prompt file SHA-256: `e31c27a2078b0b2999452ef12abeb39d690d8721446e2f777502857a7ea22577`.
- `G=8`; each timed generation was forced to 128 new tokens after an 8-token
  warmup at the same physical batch size.

| Physical generation batch | Measured throughput |
| ---: | ---: |
| 1 | 10.8 tokens/s |
| 2 | 23.4 tokens/s |
| 4 | 44.9 tokens/s |
| 8 | 77.9 tokens/s |

The probe selected **8** as the highest-throughput tested generation batch.
This does not establish that batch 8 fits a QLoRA training backward pass, nor
did the probe itself change the profile's then-current manual default of 1.
The profile now requests a fresh auto-probe (`0=auto`) when the standalone
probe is invoked; the future training runner has not been connected yet.
The container used
reference PyTorch fallbacks because `causal_conv1d` and
`flash-linear-attention` were not installed, so these speeds are specific to
this verified software stack and prompt/output length.
