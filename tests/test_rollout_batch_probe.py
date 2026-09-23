import sys
from pathlib import Path

from alignment_stress_lab.rollout_batch_probe import main


def test_manual_profile_does_not_load_model(monkeypatch, capsys, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        (root / "configs/models/qwen3_5_9b.yaml")
        .read_text(encoding="utf-8")
        .replace("physical_rollout_batch_size: 0", "physical_rollout_batch_size: 1"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["rollout_batch_probe", "--profile", str(profile)])
    main()
    output = capsys.readouterr().out
    assert "Configured manual physical batch size: 1" in output
    assert "No model loaded" in output
