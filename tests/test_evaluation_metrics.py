from alignment_stress_lab.evaluation import JudgeLabel, Verdict, summarize_labels


def label(verdict: Verdict) -> JudgeLabel:
    return JudgeLabel(
        verdict=verdict,
        evaluator="test-judge",
        protocol_version="test-v1",
    )


def test_unknown_labels_do_not_change_asr_denominator() -> None:
    summary = summarize_labels(
        [
            label(Verdict.COMPLIANCE),
            label(Verdict.REFUSAL),
            label(Verdict.UNKNOWN),
            label(Verdict.UNKNOWN),
        ]
    )

    assert summary.attack_success_rate == 0.5
    assert summary.judge_coverage == 0.5
    assert summary.unknown == 2


def test_invalid_generation_never_counts_as_success() -> None:
    summary = summarize_labels(
        [label(Verdict.COMPLIANCE), label(Verdict.INVALID)]
    )

    assert summary.attack_success_rate == 0.5
    assert summary.compliant == 1
    assert summary.invalid == 1
    assert summary.known == 2


def test_invalid_is_resolved_failure_while_unknown_is_excluded() -> None:
    summary = summarize_labels(
        [label(Verdict.UNKNOWN), label(Verdict.INVALID)]
    )

    assert summary.attack_success_rate == 0.0
    assert summary.judge_coverage == 0.5
