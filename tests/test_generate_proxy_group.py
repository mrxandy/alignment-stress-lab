from alignment_stress_lab.generate_proxy_group import completion_tokens


def test_completion_keeps_first_eos_and_drops_batch_padding() -> None:
    assert completion_tokens([11, 12, 99, 99, 99], {99}, 5) == ([11, 12, 99], "eos")


def test_completion_marks_full_length_without_eos() -> None:
    assert completion_tokens([11, 12, 13], {99}, 3) == ([11, 12, 13], "length")


def test_completion_marks_unexpected_short_stop_invalid() -> None:
    assert completion_tokens([11, 12], {99}, 3) == ([11, 12], "invalid")


def test_completion_never_exceeds_configured_cap() -> None:
    assert completion_tokens([11, 12, 13, 99], {99}, 3) == ([11, 12, 13], "length")
