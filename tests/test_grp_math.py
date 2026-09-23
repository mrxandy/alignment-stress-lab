import math
from pathlib import Path

import pytest
import torch
import yaml

from alignment_stress_lab.grp import (
    completion_token_mask,
    group_advantages,
    intent_drift_reward,
    loss_terms,
)


def test_intent_drift_reward_bounds_and_gate() -> None:
    align = torch.tensor([[10.0, 0.0, 5.0]])
    risk = torch.tensor([[10.0, 10.0, 10.0]])
    detail = torch.tensor([[10.0, 10.0, 0.0]])
    assert torch.allclose(intent_drift_reward(align, risk, detail), torch.tensor([[1, 0, 0.25]]))
    with pytest.raises(ValueError, match="within"):
        intent_drift_reward(align, risk, torch.tensor([[11.0, 0.0, 0.0]]))


def test_advantages_are_per_group_population_standardized() -> None:
    rewards = torch.tensor([[0.0, 1.0], [5.0, 5.0]], requires_grad=True)
    advantages = group_advantages(rewards)
    assert torch.allclose(advantages, torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
    assert not advantages.requires_grad
    with pytest.raises(ValueError, match="finite"):
        group_advantages(torch.tensor([[0.0, float("nan")]]))


def test_paper_loss_uses_completion_logprobs_and_positive_advantages_only() -> None:
    policy = torch.tensor(
        [[[-1.0, -2.0, float("nan")], [-3.0, 0.0, float("nan")]]],
        requires_grad=True,
    )
    reference = torch.tensor([[[-1.0, -2.0, 0.0], [-3.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    terms = loss_terms(policy, reference, torch.tensor([[1.0, -1.0]]), mask, beta=0)
    assert terms.policy.item() == pytest.approx(1.5)
    assert terms.total.item() == pytest.approx(1.5)
    assert terms.positive_rollouts == 1
    assert terms.completion_tokens == 3
    terms.total.backward()
    assert torch.allclose(policy.grad[0, 0, :2], torch.tensor([-0.5, -0.5]))
    assert policy.grad[0, 1, 0].item() == 0
    assert torch.all(policy.grad[~mask] == 0)


def test_kl_applies_without_positive_advantage_and_has_gradient() -> None:
    policy = torch.tensor([[[-1.0], [-1.0]]], requires_grad=True)
    reference = torch.tensor([[[-0.5], [-1.0]]], requires_grad=True)
    old = torch.tensor([[[-1.0], [-1.0]]], requires_grad=True)
    mask = torch.ones_like(policy, dtype=torch.bool)
    terms = loss_terms(
        policy,
        reference,
        torch.zeros((1, 2)),
        mask,
        beta=0.3,
        mode="clipped_ratio",
        old_logp=old,
    )
    assert terms.policy.item() == 0
    assert terms.kl.item() == pytest.approx((math.exp(0.5) - 1.5) / 2)
    terms.total.backward()
    assert policy.grad[0, 0, 0].item() != 0
    assert reference.grad is None
    assert old.grad is None


def test_clipped_ratio_caps_positive_policy_gradient() -> None:
    policy = torch.tensor([[[-1.0 + math.log(2)], [-1.0]]], requires_grad=True)
    old = torch.tensor([[[-1.0], [-1.0]]])
    reference = policy.detach().clone()
    mask = torch.ones_like(policy, dtype=torch.bool)
    terms = loss_terms(
        policy,
        reference,
        torch.tensor([[1.0, 0.0]]),
        mask,
        beta=0,
        mode="clipped_ratio",
        old_logp=old,
        clip_epsilon=0.2,
    )
    assert terms.policy.item() == pytest.approx(-0.6)
    terms.total.backward()
    assert policy.grad[0, 0, 0].item() == 0
    assert policy.grad[0, 1, 0].item() == 0


def test_current_equals_old_has_same_policy_gradient_as_paper_form() -> None:
    old = torch.tensor([[[-1.0], [-2.0]]])
    mask = torch.ones_like(old, dtype=torch.bool)
    gradients = []
    for mode in ("paper_logprob", "clipped_ratio"):
        current = old.detach().clone().requires_grad_()
        kwargs = {"mode": mode}
        if mode == "clipped_ratio":
            kwargs["old_logp"] = old
        terms = loss_terms(current, old, torch.tensor([[1.0, -1.0]]), mask, beta=0, **kwargs)
        terms.total.backward()
        gradients.append(current.grad.clone())
    assert torch.allclose(gradients[0], gradients[1])


def test_extreme_positive_log_ratio_clips_without_overflow() -> None:
    current = torch.tensor([[[-1.0], [-1.0]]], requires_grad=True)
    old = torch.tensor([[[-1001.0], [-1.0]]])
    mask = torch.ones_like(current, dtype=torch.bool)
    terms = loss_terms(
        current,
        current.detach(),
        torch.tensor([[1.0, 0.0]]),
        mask,
        beta=0,
        mode="clipped_ratio",
        old_logp=old,
    )
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert current.grad[0, 0, 0].item() == 0


def test_positive_only_lower_clip_does_not_inflate_a_small_ratio() -> None:
    current = torch.tensor([[[-1.0 - math.log(10)], [-1.0]]])
    old = torch.tensor([[[-1.0], [-1.0]]])
    mask = torch.ones_like(current, dtype=torch.bool)
    terms = loss_terms(
        current,
        current,
        torch.tensor([[1.0, 0.0]]),
        mask,
        beta=0,
        mode="clipped_ratio",
        old_logp=old,
        clip_epsilon=0.2,
    )
    assert terms.policy.item() == pytest.approx(-0.05)


def test_loss_rejects_single_rollout_group() -> None:
    logp = torch.tensor([[[-1.0]]])
    with pytest.raises(ValueError, match="G >= 2"):
        loss_terms(
            logp,
            logp,
            torch.tensor([[0.0]]),
            torch.ones_like(logp, dtype=torch.bool),
            beta=0,
        )


def test_eos_included_but_prompt_and_eos_padding_excluded() -> None:
    ids = torch.tensor([[[42, 9, 2, 2, 2], [42, 9, 10, 11, 2]]])
    mask = completion_token_mask(
        ids,
        torch.tensor([[1, 1]]),
        torch.tensor([[2, 3]]),
        [["eos", "length"]],
        eos_token_id=2,
        max_completion_tokens=3,
    )
    assert mask.tolist() == [[[False, True, True, False, False], [False, True, True, True, False]]]


def test_invalid_record_has_no_loss_tokens_and_bad_eos_is_rejected() -> None:
    ids = torch.tensor([[[42, 9, 2], [42, 9, 2]]])
    mask = completion_token_mask(
        ids,
        torch.tensor([[1, 1]]),
        torch.tensor([[2, 0]]),
        [["eos", "invalid"]],
        eos_token_id=2,
        max_completion_tokens=2,
    )
    assert mask.tolist() == [[[False, True, True], [False, False, False]]]
    with pytest.raises(ValueError, match="first generated EOS"):
        completion_token_mask(
            torch.tensor([[[42, 2, 2]]]),
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            [["eos"]],
            eos_token_id=2,
            max_completion_tokens=2,
        )
    with pytest.raises(ValueError, match="integer dtypes"):
        completion_token_mask(
            ids,
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[2, 0]]),
            [["eos", "invalid"]],
            eos_token_id=2,
            max_completion_tokens=2,
        )


def test_both_dense_profiles_use_the_same_group_math() -> None:
    root = Path(__file__).resolve().parents[1]
    profiles = [
        yaml.safe_load((root / "configs" / "models" / name).read_text(encoding="utf-8"))
        for name in ("qwen3_5_9b.yaml", "qwen3_8_27b.yaml")
    ]
    assert all(profile["architecture"] == "dense" for profile in profiles)
    assert all(profile["training"]["logical_group_size"] == 8 for profile in profiles)
    rewards = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    advantages = group_advantages(rewards)
    assert advantages.shape == (2, 8)
    assert torch.allclose(advantages[0], advantages[1])
