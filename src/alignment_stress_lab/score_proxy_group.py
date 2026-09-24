"""Score one saved fictional proxy group with the external three-score Judge."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from alignment_stress_lab.judge import JudgeConfig, JudgeInput
from alignment_stress_lab.judge.proxy_client import ProxyJudgeClient, ProxyScore


def proxy_group_rows(lines: Iterable[str], group_size: int) -> list[dict[str, object]]:
    """Validate row order and the minimum data needed for safe scoring."""

    if type(group_size) is not int or group_size < 2:
        raise ValueError("Manifest group_size must be an integer >= 2")
    rows: list[dict[str, object]] = []
    prompt = None
    for text in lines:
        row = json.loads(text)
        if (
            not isinstance(row, dict)
            or type(row.get("rollout_index")) is not int
            or row["rollout_index"] != len(rows)
        ):
            raise ValueError("Proxy rollout indices must be consecutive from zero")
        if type(row.get("prompt")) is not str or not row["prompt"].strip():
            raise ValueError("Proxy rollout prompt is missing")
        if prompt is None:
            prompt = row["prompt"]
        elif row["prompt"] != prompt:
            raise ValueError("Proxy rollouts must share exactly one prompt")
        if type(row.get("response")) is not str:
            raise ValueError("Proxy rollout response is missing")
        if row.get("finish_reason") not in ("eos", "length", "invalid"):
            raise ValueError("Proxy rollout has invalid finish_reason")
        token_ids = row.get("completion_token_ids")
        if not isinstance(token_ids, list) or any(type(value) is not int for value in token_ids):
            raise ValueError("Proxy rollout has invalid completion_token_ids")
        if type(row.get("generated_tokens")) is not int or row["generated_tokens"] != len(
            token_ids
        ):
            raise ValueError("Proxy rollout token count does not match IDs")
        rows.append(row)
    if len(rows) != group_size:
        raise ValueError("Proxy rollout count does not match manifest group_size")
    return rows


def _manifest(input_path: Path) -> dict[str, object]:
    manifest_path = input_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Proxy manifest must be an object")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if manifest.get("output_sha256") != digest:
        raise ValueError("Proxy rollout JSONL SHA-256 does not match its manifest")
    return manifest


async def _score(
    rows: list[dict[str, object]], config: JudgeConfig
) -> tuple[list[ProxyScore], str]:
    items = [
        JudgeInput(
            str(row["prompt"]),
            str(row["response"]) if row["finish_reason"] == "eos" else "",
        )
        for row in rows
    ]
    async with ProxyJudgeClient(config) as judge:
        scores = await judge.evaluate_group(items)
        protocol_version = judge.protocol_version
    if scores is None:
        raise RuntimeError("Proxy Judge result unresolved; no score output was written")
    return scores, protocol_version


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/judge.proxy.deepseek.yaml")
    )
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Score output or summary exists; choose a new --output")
    manifest = _manifest(args.input)
    with args.input.open(encoding="utf-8") as source:
        rows = proxy_group_rows(source, manifest.get("group_size"))
    config = JudgeConfig.from_yaml(args.config)
    scores, protocol_version = await _score(rows, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        for row, score in zip(rows, scores, strict=True):
            output.write(
                json.dumps(
                    {
                        "rollout_index": row["rollout_index"],
                        "status": (
                            "scored"
                            if row["finish_reason"] == "eos" and str(row["response"]).strip()
                            else "invalid"
                        ),
                        "fiction_marking": score.fiction_marking,
                        "instruction_following": score.instruction_following,
                        "completeness": score.completeness,
                        "eligible": score.eligible,
                        "reward": score.reward,
                        "evaluator": config.model,
                        "protocol_version": protocol_version,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "input_sha256": manifest["output_sha256"],
        "evaluator": config.model,
        "protocol_version": protocol_version,
        "total": len(scores),
        "invalid_generations": sum(
            row["finish_reason"] != "eos" or not str(row["response"]).strip()
            for row in rows
        ),
        "eligible": sum(score.eligible for score in scores),
        "reward_mean": sum(score.reward for score in scores) / len(scores),
        "judge_coverage": 1.0,
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"scored {len(scores)} proxy rollouts; output={args.output}")
    print(f"eligible={summary['eligible']} invalid={summary['invalid_generations']}")
    print(f"reward_mean={summary['reward_mean']:.3f}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
