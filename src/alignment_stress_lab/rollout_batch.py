"""Choose a generation batch size from measured throughput within one group."""

from __future__ import annotations

import math
from collections.abc import Callable


def candidate_batch_sizes(group_size: int) -> list[int]:
    """Try powers of two and the group-size endpoint, never exceeding G."""

    if type(group_size) is not int or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    sizes = []
    batch_size = 1
    while batch_size < group_size:
        sizes.append(batch_size)
        batch_size *= 2
    sizes.append(group_size)
    return sizes


def choose_rollout_batch_size(
    group_size: int, probe: Callable[[int], float | None]
) -> int:
    """Choose the highest-throughput feasible size; None means a failed probe.

    The caller must benchmark actual model generation with representative
    prompt/output lengths and catch only known resource failures. This selector
    does not probe training/backward memory or call the model itself.
    """

    best_size = 0
    best_throughput = 0.0
    for batch_size in candidate_batch_sizes(group_size):
        throughput = probe(batch_size)
        if throughput is None:
            if batch_size == 1:
                raise RuntimeError("Generation fails even at physical batch size 1")
            break
        if not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("Probe throughput must be finite and positive")
        if throughput > best_throughput:
            best_size = batch_size
            best_throughput = throughput
    return best_size
