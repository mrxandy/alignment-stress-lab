import pytest

from alignment_stress_lab.rollout_batch import candidate_batch_sizes, choose_rollout_batch_size


def test_candidates_stop_at_g() -> None:
    assert candidate_batch_sizes(8) == [1, 2, 4, 8]
    assert candidate_batch_sizes(6) == [1, 2, 4, 6]


def test_selects_best_throughput_not_largest_batch() -> None:
    scores = {1: 10.0, 2: 22.0, 4: 42.0, 8: 35.0}
    assert choose_rollout_batch_size(8, scores.get) == 4


def test_stops_after_failed_probe() -> None:
    attempted = []

    def probe(size: int) -> float | None:
        attempted.append(size)
        return {1: 10.0, 2: 22.0, 4: None}[size]

    assert choose_rollout_batch_size(8, probe) == 2
    assert attempted == [1, 2, 4]


def test_size_one_failure_is_fatal() -> None:
    with pytest.raises(RuntimeError, match="size 1"):
        choose_rollout_batch_size(8, lambda _: None)


@pytest.mark.parametrize("score", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_throughput_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="throughput"):
        choose_rollout_batch_size(8, lambda _: score)
