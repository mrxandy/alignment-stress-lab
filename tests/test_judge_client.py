import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from alignment_stress_lab.evaluation import Verdict, summarize_labels
from alignment_stress_lab.judge import JudgeClient, JudgeConfig, JudgeInput


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


def fake_client(*replies):
    completions = FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def config(**overrides):
    values = {"model": "offline-test-judge", "cache_path": Path(":memory:")}
    values.update(overrides)
    return JudgeConfig(**values)


def test_valid_labels_preserve_order_and_cache() -> None:
    client, calls = fake_client(json.dumps({"labels": ["refusal", "compliance"]}))

    async def run():
        async with JudgeClient(config(), client=client) as judge:
            items = [JudgeInput("request one", "I decline."), JudgeInput("request two", "Done.")]
            first = await judge.evaluate(items)
            second = await judge.evaluate(items)
            return first, second

    first, second = asyncio.run(run())
    assert [x.verdict for x in first] == [Verdict.REFUSAL, Verdict.COMPLIANCE]
    assert first == second
    assert calls.calls == 1


def test_empty_response_is_invalid_without_api_request() -> None:
    client, calls = fake_client()

    async def run():
        async with JudgeClient(config(), client=client) as judge:
            return await judge.evaluate([JudgeInput("request", "  ")])

    labels = asyncio.run(run())
    assert labels[0].verdict is Verdict.INVALID
    assert summarize_labels(labels).attack_success_rate == 0.0
    assert calls.calls == 0


def test_malformed_batch_becomes_unknown_and_is_not_cached() -> None:
    client, calls = fake_client(
        json.dumps({"labels": ["compliance", "refusal"]}),
        json.dumps({"labels": ["compliance"]}),
    )

    async def run():
        async with JudgeClient(config(max_attempts=1), client=client) as judge:
            item = [JudgeInput("request", "response")]
            first = await judge.evaluate(item)
            second = await judge.evaluate(item)
            return first, second

    first, second = asyncio.run(run())
    assert first[0].verdict is Verdict.UNKNOWN
    assert second[0].verdict is Verdict.COMPLIANCE
    assert calls.calls == 2

def test_network_failure_stays_unknown_after_retries() -> None:
    client, calls = fake_client(OSError("offline"), OSError("offline"))

    async def run():
        async with JudgeClient(config(max_attempts=2), client=client) as judge:
            return await judge.evaluate([JudgeInput("request", "response")])

    labels = asyncio.run(run())
    assert labels[0].verdict is Verdict.UNKNOWN
    assert summarize_labels(labels).attack_success_rate is None
    assert calls.calls == 2
