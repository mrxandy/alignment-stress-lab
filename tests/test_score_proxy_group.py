import hashlib
import json
from pathlib import Path

import pytest

from alignment_stress_lab.score_proxy_group import _manifest, proxy_group_rows


def row(index: int, prompt: str = "fiction") -> dict:
    return {
        "rollout_index": index,
        "prompt": prompt,
        "response": "fictional response",
        "completion_token_ids": [11, 99],
        "generated_tokens": 2,
        "finish_reason": "eos",
    }


def test_valid_group_has_exactly_g_ordered_rows() -> None:
    lines = [json.dumps(row(0)), json.dumps(row(1))]
    assert len(proxy_group_rows(lines, 2)) == 2


@pytest.mark.parametrize(
    "rows",
    [
        [row(0)],
        [row(0), row(0)],
        [row(0), row(1, "different")],
        [row(0), {**row(1), "generated_tokens": 3}],
    ],
)
def test_bad_group_is_rejected(rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        proxy_group_rows([json.dumps(item) for item in rows], 2)


def test_manifest_requires_matching_response_hash(tmp_path: Path) -> None:
    responses = tmp_path / "group.jsonl"
    responses.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "group.manifest.json"
    digest = hashlib.sha256(responses.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"output_sha256": digest}), encoding="utf-8")
    assert _manifest(responses)["output_sha256"] == digest
    responses.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _manifest(responses)
