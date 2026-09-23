from pathlib import Path, PurePosixPath

import pytest

from alignment_stress_lab.model_smoke import model_path_from_profile


def test_9b_profile_points_to_local_aligned_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = root / "configs/models/qwen3_5_9b.yaml"
    assert model_path_from_profile(profile) == PurePosixPath(
        "/workspace/models/modelscope/Qwen/Qwen3.5-9B"
    )


def test_27b_profile_is_not_silently_selected() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = root / "configs/models/qwen3_8_27b.yaml"
    with pytest.raises(ValueError, match="Qwen3.5-9B"):
        model_path_from_profile(profile)
