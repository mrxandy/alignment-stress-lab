import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignment_stress_lab.judge import JudgeConfig, JudgeInput
from alignment_stress_lab.judge.proxy_client import ProxyJudgeClient, ProxyScore, _parse_scores


class FakeCompletions:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def fake_client(*replies):
    completions = FakeCompletions(*replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def config(**overrides):
    values = {"model": "offline-test-judge", "cache_path": Path(":memory:")}
    values.update(overrides)
    return JudgeConfig(**values)


def score(marking=10, following=8, completeness=6, eligible=True):
    return {
        "fiction_marking": marking,
        "instruction_following": following,
        "completeness": completeness,
        "eligible": eligible,
    }


def test_valid_scores_preserve_order_and_cache() -> None:
    client, calls = fake_client(json.dumps({"scores": [score(), score(eligible=False)]}))

    async def run():
        async with ProxyJudgeClient(config(), client=client) as judge:
            items = [JudgeInput("fiction", "first"), JudgeInput("fiction", "second")]
            first = await judge.evaluate_group(items)
            second = await judge.evaluate_group(items)
            return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert first is not None
    assert first[0].reward == 0.7
    assert first[1].reward == 0.0
    assert calls.calls == 1


def test_empty_response_gets_zero_without_api_request() -> None:
    client, calls = fake_client()

    async def run():
        async with ProxyJudgeClient(config(), client=client) as judge:
            return await judge.evaluate_group([JudgeInput("p", ""), JudgeInput("p", " ")])

    scores = asyncio.run(run())
    assert scores == [ProxyScore(0, 0, 0, False), ProxyScore(0, 0, 0, False)]
    assert calls.calls == 0


def test_malformed_batch_is_unknown_and_not_cached() -> None:
    client, calls = fake_client(
        json.dumps({"scores": [score()]}),
        json.dumps({"scores": [score(), score()]}),
    )

    async def run():
        async with ProxyJudgeClient(config(max_attempts=1), client=client) as judge:
            items = [JudgeInput("p", "first"), JudgeInput("p", "second")]
            return await judge.evaluate_group(items), await judge.evaluate_group(items)

    first, second = asyncio.run(run())
    assert first is None
    assert second is not None
    assert calls.calls == 2


def test_network_failure_keeps_whole_group_unknown() -> None:
    client, calls = fake_client(OSError("offline"), OSError("offline"))

    async def run():
        async with ProxyJudgeClient(config(max_attempts=2), client=client) as judge:
            return await judge.evaluate_group([JudgeInput("p", "first"), JudgeInput("p", "second")])

    assert asyncio.run(run()) is None
    assert calls.calls == 2


def test_mixed_prompt_group_is_rejected_without_api_request() -> None:
    client, calls = fake_client()

    async def run():
        async with ProxyJudgeClient(config(), client=client) as judge:
            return await judge.evaluate_group(
                [JudgeInput("first prompt", "a"), JudgeInput("second prompt", "b")]
            )

    with pytest.raises(ValueError, match="one non-empty prompt"):
        asyncio.run(run())
    assert calls.calls == 0


@pytest.mark.parametrize(
    "bad",
    [score(marking=True), score(following=11), score(completeness=-1), score(eligible=1)],
)
def test_invalid_score_fields_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        _parse_scores({"scores": [bad]}, 1)
