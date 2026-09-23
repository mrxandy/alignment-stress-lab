"""Completion-only token mask with explicit EOS and length-stop semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

FinishReason = Literal["eos", "length", "invalid"]


def completion_token_mask(
    input_ids: Tensor,
    prompt_lengths: Tensor,
    completion_lengths: Tensor,
    finish_reasons: Sequence[Sequence[FinishReason]],
    *,
    eos_token_id: int,
    max_completion_tokens: int,
) -> Tensor:
    """Return [B, G, T] mask aligned with scored token IDs.

    The generated EOS token is included. Padding is excluded by explicit
    lengths, even when ``pad_token_id == eos_token_id``. A length stop keeps all
    generated tokens; an invalid/incomplete record contributes none.
    """

    if input_ids.ndim != 3:
        raise ValueError("input_ids must have shape [B, G, T]")
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    lengths_and_ids = (input_ids, prompt_lengths, completion_lengths)
    if any(tensor.dtype not in integer_dtypes for tensor in lengths_and_ids):
        raise ValueError("Token IDs and lengths must use integer dtypes")
    shape = input_ids.shape[:2]
    if prompt_lengths.shape != shape or completion_lengths.shape != shape:
        raise ValueError("Length tensors must have shape [B, G]")
    if len(finish_reasons) != shape[0] or any(len(row) != shape[1] for row in finish_reasons):
        raise ValueError("finish_reasons must have shape [B, G]")
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive")

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch in range(shape[0]):
        for rollout in range(shape[1]):
            start = int(prompt_lengths[batch, rollout])
            length = int(completion_lengths[batch, rollout])
            reason = finish_reasons[batch][rollout]
            if start < 1 or length < 0 or length > max_completion_tokens:
                raise ValueError("Invalid prompt or completion length")
            if start + length > input_ids.shape[-1]:
                raise ValueError("Completion extends past input_ids")
            if reason not in ("eos", "length", "invalid"):
                raise ValueError("Unknown finish reason")
            if reason == "invalid":
                continue
            if length == 0:
                raise ValueError("Completed rollout has no generated tokens")
            tokens = input_ids[batch, rollout, start : start + length]
            if reason == "eos":
                if int(tokens[-1]) != eos_token_id or (tokens[:-1] == eos_token_id).any():
                    raise ValueError("EOS stop must end at the first generated EOS")
            elif length != max_completion_tokens or (tokens == eos_token_id).any():
                raise ValueError("Length stop must reach the cap without an EOS")
            mask[batch, rollout, start : start + length] = True
    return mask
