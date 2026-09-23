import asyncio
import json
from types import SimpleNamespace

import pytest

from alignment_stress_lab.evaluation import JudgeLabel, Verdict
from alignment_stress_lab.score_baseline import baseline_rows, score_rows


def test_baseline_rows_validate_ids_and_finish_reasons() -> None:
    first = {"line_number": 1, "prompt": "question", "response": "answer", "finish_reason": "eos"}
    second = {"line_number": 3, "prompt": "question", "response": "", "finish_reason": "length"}
    assert [row["line_number"] for row in baseline_rows(map(json.dumps, (first, second)))] == [
        1,
        3,
    ]
    with pytest.raises(ValueError, match="repeated"):
        baseline_rows(map(json.dumps, (first, first)))


def test_length_stop_is_invalid_without_calling_judge() -> None:
    class FakeJudge:
        config = SimpleNamespace(model="fake")
        protocol_version = "v1"

        async def evaluate(self, inputs):
            assert len(inputs) == 1
            return [JudgeLabel(Verdict.REFUSAL, "fake", "v1")]

    rows = [
        {"line_number": 1, "prompt": "one", "response": "no", "finish_reason": "eos"},
        {"line_number": 2, "prompt": "two", "response": "partial", "finish_reason": "length"},
    ]
    labels = asyncio.run(score_rows(rows, FakeJudge()))
    assert [label.verdict for label in labels] == [Verdict.REFUSAL, Verdict.INVALID]
