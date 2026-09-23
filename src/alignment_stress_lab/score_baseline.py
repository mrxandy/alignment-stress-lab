"""Score baseline JSONL with the configured external Judge API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from alignment_stress_lab.evaluation import JudgeLabel, Verdict, summarize_labels
from alignment_stress_lab.judge import JudgeClient, JudgeConfig, JudgeInput


def baseline_rows(lines: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for file_line, text in enumerate(lines, start=1):
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"Baseline line {file_line} is not an object")
        line_number = row.get("line_number")
        if not isinstance(line_number, int) or line_number < 1 or line_number in seen:
            raise ValueError(f"Invalid or repeated source line number at baseline line {file_line}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Missing prompt at baseline line {file_line}")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"Missing response at baseline line {file_line}")
        if row.get("finish_reason") not in ("eos", "length"):
            raise ValueError(f"Invalid finish reason at baseline line {file_line}")
        seen.add(line_number)
        rows.append(row)
    if not rows:
        raise ValueError("Baseline JSONL is empty")
    return rows


async def score_rows(
    rows: list[dict[str, object]], judge: JudgeClient
) -> list[JudgeLabel]:
    inputs = [
        JudgeInput(str(row["prompt"]), str(row["response"]))
        for row in rows
        if row["finish_reason"] == "eos"
    ]
    judged = iter(await judge.evaluate(inputs))
    labels = []
    for row in rows:
        if row["finish_reason"] == "eos":
            labels.append(next(judged))
        else:
            labels.append(
                JudgeLabel(
                    Verdict.INVALID,
                    judge.config.model,
                    judge.protocol_version,
                    "generation reached token limit",
                )
            )
    return labels


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/judge.example.yaml"))
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Score output or summary exists; choose a new --output")
    with args.input.open(encoding="utf-8") as source:
        rows = baseline_rows(source)
    config = JudgeConfig.from_yaml(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with JudgeClient(config) as judge:
        labels = await score_rows(rows, judge)
        protocol_version = judge.protocol_version

    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        for row, label in zip(rows, labels, strict=True):
            output.write(
                json.dumps(
                    {
                        "line_number": row["line_number"],
                        "verdict": label.verdict.value,
                        "evaluator": label.evaluator,
                        "protocol_version": label.protocol_version,
                        "reason": label.reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = summarize_labels(labels)
    resolved = summary.compliant + summary.refused + summary.invalid
    with summary_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(
            {
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "evaluator": config.model,
                "protocol_version": protocol_version,
                "total": summary.total,
                "compliant": summary.compliant,
                "refused": summary.refused,
                "invalid": summary.invalid,
                "unknown": summary.unknown,
                "judge_coverage": summary.judge_coverage,
                "refusal_rate": summary.refused / resolved if resolved else None,
                "attack_success_rate": summary.attack_success_rate,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")
    print(f"scored {summary.total} rows; coverage={summary.judge_coverage:.1%}")
    print(f"refusal_rate={summary.refused / resolved if resolved else None}")
    print(f"ASR={summary.attack_success_rate}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
