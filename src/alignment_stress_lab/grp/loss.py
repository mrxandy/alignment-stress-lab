"""Differentiable GRP-Oblit objective on already aligned token log-probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

LossMode = Literal["paper_logprob", "clipped_ratio"]


@dataclass(frozen=True, slots=True)
class LossTerms:
    total: Tensor
    policy: Tensor
    kl: Tensor
    positive_rollouts: int
    completion_tokens: int


def intent_drift_reward(align: Tensor, risk: Tensor, detail: Tensor) -> Tensor:
    """Paper's R_IDA = align * (risk + detail) / 200, normalized to [0, 1]."""

    if align.shape != risk.shape or align.shape != detail.shape:
        raise ValueError("All reward components must have the same shape")
    for component in (align, risk, detail):
        if not torch.isfinite(component).all() or ((component < 0) | (component > 10)).any():
            raise ValueError("Reward components must be finite and within [0, 10]")
    return align.detach() * (risk.detach() + detail.detach()) / 200.0


def group_advantages(rewards: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    """Standardize rewards separately for every prompt group [B, G]."""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Rewards must have shape [prompts, rollouts] with rollouts >= 2")
    if epsilon <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("epsilon must be positive and all rewards must be finite")
    r = rewards.detach().float()
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, correction=0)
    return (r - mean) / (std + epsilon)


def loss_terms(
    policy_logp: Tensor,
    reference_logp: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
    *,
    beta: float,
    mode: LossMode = "paper_logprob",
    old_logp: Tensor | None = None,
    clip_epsilon: float = 0.2,
) -> LossTerms:
    """Calculate the response-level objective averaged over every rollout.

    Shapes: log-probabilities and mask [B, G, T], advantages [B, G].
    Token log-probabilities must already be shifted to the matching token IDs.
    ``paper_logprob`` follows the paper's sequence log-probability expression.
    ``clipped_ratio`` is an explicit off-policy extension for reused rollouts.
    """

    if policy_logp.ndim != 3 or policy_logp.shape[0] < 1 or policy_logp.shape[1] < 2:
        raise ValueError("Log-probabilities must have shape [B, G, T] with B >= 1, G >= 2")
    if reference_logp.shape != policy_logp.shape or completion_mask.shape != policy_logp.shape:
        raise ValueError("Reference log-probabilities and mask must match policy shape")
    if advantages.shape != policy_logp.shape[:2]:
        raise ValueError("Advantages must have shape [B, G]")
    if completion_mask.dtype is not torch.bool:
        raise ValueError("completion_mask must have boolean dtype")
    if beta < 0 or not torch.isfinite(torch.tensor(beta)):
        raise ValueError("beta must be finite and nonnegative")
    if mode not in ("paper_logprob", "clipped_ratio"):
        raise ValueError("Unknown loss mode")
    if mode == "clipped_ratio":
        if old_logp is None or old_logp.shape != policy_logp.shape:
            raise ValueError("clipped_ratio requires aligned old_logp")
        if clip_epsilon <= 0 or not torch.isfinite(torch.tensor(clip_epsilon)):
            raise ValueError("clip_epsilon must be finite and positive")
    elif old_logp is not None:
        raise ValueError("old_logp is only used by clipped_ratio")

    active = completion_mask
    if not active.any():
        raise ValueError("At least one completion token is required")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite")
    if not torch.isfinite(policy_logp[active]).all():
        raise ValueError("Active policy log-probabilities must be finite")
    if not torch.isfinite(reference_logp[active]).all():
        raise ValueError("Active reference log-probabilities must be finite")
    if old_logp is not None and not torch.isfinite(old_logp[active]).all():
        raise ValueError("Active old log-probabilities must be finite")

    # Cast before expm1 for stable KL under bf16/fp16 model execution.
    policy = torch.where(active, policy_logp.float(), 0.0)
    reference = torch.where(active, reference_logp.detach().float(), 0.0)
    advantage = advantages.detach().float().clamp_min(0).unsqueeze(-1)
    mask = active.to(policy.dtype)
    denominator = policy.shape[0] * policy.shape[1]

    if mode == "paper_logprob":
        surrogate = policy
    else:
        old = torch.where(active, old_logp.detach().float(), 0.0)
        # For A > 0, min(ratio*A, clipped_ratio*A) equals
        # min(ratio, 1+epsilon)*A. Clip log-ratio before exp to avoid overflow.
        surrogate = (policy - old).clamp(max=math.log1p(clip_epsilon)).exp()

    policy_loss = -(surrogate * advantage * mask).sum() / denominator
    # k3 = exp(ref - policy) - (ref - policy) - 1 is a nonnegative
    # sampled-token estimator of KL(policy || reference), not full-vocab KL.
    delta = reference - policy
    sampled_kl = torch.expm1(delta) - delta
    kl_loss = (sampled_kl * mask).sum() / denominator
    total = policy_loss + beta * kl_loss
    if not torch.isfinite(total):
        raise ValueError("Loss is non-finite")
    positive = int(((advantages > 0) & active.any(dim=-1)).sum().item())
    return LossTerms(total, policy_loss, kl_loss, positive, int(active.sum().item()))
