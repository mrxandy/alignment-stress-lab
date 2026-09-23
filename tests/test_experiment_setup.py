from pathlib import Path

import pytest

from alignment_stress_lab.experiment_setup import load_experiment_setup


def test_default_9b_proxy_reads_configured_g() -> None:
    root = Path(__file__).resolve().parents[1]
    setup = load_experiment_setup(
        root / "configs/models/qwen3_5_9b.yaml",
        root / "data/prompts/fictional_news_train_proxy_zh.txt",
    )
    assert setup.group_size == 8
    assert setup.physical_rollout_batch_size == 0
    assert setup.rollout_max_new_tokens == 128
    assert setup.prompt
    assert len(setup.prompt_sha256) == 64


def test_configured_g_can_change_without_code_change(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace("logical_group_size: 8", "logical_group_size: 4"),
        encoding="utf-8",
    )
    setup = load_experiment_setup(
        profile, root / "data/prompts/fictional_news_train_proxy_zh.txt"
    )
    assert setup.group_size == 4


@pytest.mark.parametrize("value", ["1", "true", "4.5"])
def test_invalid_g_is_rejected(tmp_path: Path, value: str) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace("logical_group_size: 8", f"logical_group_size: {value}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="logical_group_size"):
        load_experiment_setup(
            profile, root / "data/prompts/fictional_news_train_proxy_zh.txt"
        )


def test_multiple_prompts_are_rejected_for_single_prompt_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_experiment_setup(root / "configs/models/qwen3_5_9b.yaml", prompts)


@pytest.mark.parametrize("value", ["1", "2"])
def test_physical_batch_accepts_auto_and_manual(tmp_path: Path, value: str) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "physical_rollout_batch_size: 0",
            f"physical_rollout_batch_size: {value}",
        ),
        encoding="utf-8",
    )
    setup = load_experiment_setup(
        profile, root / "data/prompts/fictional_news_train_proxy_zh.txt"
    )
    assert setup.physical_rollout_batch_size == int(value)


@pytest.mark.parametrize("value", ["-1", "9", "true", "1.5"])
def test_invalid_physical_batch_is_rejected(tmp_path: Path, value: str) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "physical_rollout_batch_size: 0",
            f"physical_rollout_batch_size: {value}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="physical_rollout_batch_size"):
        load_experiment_setup(
            profile, root / "data/prompts/fictional_news_train_proxy_zh.txt"
        )


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5"])
def test_invalid_max_new_tokens_is_rejected(tmp_path: Path, value: str) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace("rollout_max_new_tokens: 128", f"rollout_max_new_tokens: {value}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rollout_max_new_tokens"):
        load_experiment_setup(
            profile, root / "data/prompts/fictional_news_train_proxy_zh.txt"
        )
