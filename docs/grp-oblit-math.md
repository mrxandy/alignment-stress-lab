# GRP-Oblit mathematical contract (v1)

Source: [GRP-Obliteration, arXiv:2602.06258v1](https://arxiv.org/abs/2602.06258),
Section 3.1, pages 3-4. This document records both the paper's formulas and
the implementation choices needed to turn them into token-level code. It is
not a claim that this project reproduces the paper's results.

## Inputs and reward

Each prompt has `G >= 2` sampled rollouts. The paper uses `G = 8` in its
reported experiments. Its Judge returns three scores in `[0, 10]`: intent
alignment, compliance risk, and detail. The scalar reward is

`R_IDA = 0.5 * alignment * (risk + detail) / 100`.

Thus all-zero intent yields zero reward and all three tens yield one. A
missing Judge score is **not** zero: the whole group must be retried or skipped
by the caller. The loss kernel rejects non-finite rewards.

## Group advantages

For each prompt separately, compute its group mean and **population** standard
deviation. With fixed `epsilon = 1e-8`:

`A_i = (R_i - mean(R_group)) / (std_population(R_group) + epsilon)`.

The paper says standard deviation without specifying the sample correction;
population standard deviation is our explicit convention. Constant-reward
groups have zero advantages. Advantages and reference values are detached from
autograd. Only `A_i > 0` receives a policy improvement term; zero and negative
advantages have no policy term. KL still applies to every valid rollout.

## Policy objective and importance ratios

The paper writes `-mean_group[1(A_i > 0) * A_i * log policy(o_i | p)]`.
`paper_logprob` implements this literally as the sum of scored completion
token log-probabilities for each rollout, followed by division by the number
of rollouts across prompt groups (`B * G`). No prompt token contributes.

The displayed paper formula does **not** define an importance ratio or a clip
threshold. `clipped_ratio` is our separately named extension for reusing
rollouts after the policy changes. For every completion token it uses
`ratio = exp(current_logp - old_logp)` and the surrogate
`min(ratio * A_positive, clip(ratio, 1-epsilon_clip, 1+epsilon_clip) * A_positive)`.
It is also averaged over `B * G`. With positive-only advantages, the lower
clip bound does not affect the minimum; the upper bound caps policy incentive.
The old log-probability is detached. Do not report this mode as the paper's
exact equation. At `current == old`, its policy gradient matches the
`paper_logprob` gradient, although its scalar loss value differs.

## KL anchor

The paper adds `beta * KL(policy || frozen_reference)` but does not specify a
token-level estimator. The implementation uses the nonnegative sampled-token
estimator `k3 = exp(reference_logp - policy_logp) -
(reference_logp - policy_logp) - 1`, summed over valid completion tokens and
divided by `B * G`. Reference log-probabilities are detached. This estimator
approximates the paper's full-distribution KL; it is not an exact full-vocab
KL. In `clipped_ratio` mode, samples were drawn from an older policy, so this
is a drift penalty rather than an unbiased current-policy KL estimate.
`beta >= 0` is a model-family-specific experimental setting, never
silently shared between 9B and 27B.

## Token alignment, EOS, and truncation

`input_ids[b,g,t]` and `policy_logp[b,g,t]` refer to the same token. The model
adapter must perform the causal shift **before** calling the loss. The mask
builder uses explicit prompt and completion lengths; it never infers padding
from token ID, because a tokenizer may use EOS as PAD.

- Prompt positions and right padding are masked out.
- For `finish_reason="eos"`, the first generated EOS must be the final active
  completion token, and it is included in the loss.
- For `finish_reason="length"`, the completion must reach the configured
  generation cap without EOS. All generated tokens are included. This is a
  normal max-token stop, distinct from a broken record.
- For `finish_reason="invalid"`, the rollout has no active loss tokens. Its
  reward must still be an explicit finite failed-attempt value if the group
  is trained; unknown Judge results cannot be converted to this state.

All arrays use `[B, G, T]` for token data and `[B, G]` for scalar rollout data.
The code contains no 9B/27B-specific branches. Model profile files select the
same group size and Dense architecture; their later resource settings can
differ without changing this mathematical contract.
