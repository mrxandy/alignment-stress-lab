import sys

from alignment_stress_lab.rollout_batch_probe import main


def test_manual_profile_does_not_load_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["rollout_batch_probe"])
    main()
    output = capsys.readouterr().out
    assert "Configured manual physical batch size: 1" in output
    assert "No model loaded" in output
