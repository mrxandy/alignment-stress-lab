"""Validate the single-prompt 9B proxy experiment without loading a model."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from alignment_stress_lab.baseline import prompt_rows
from alignment_stress_lab.model_smoke import model_path_from_profile


@dataclass(frozen=True, slots=True)
class ExperimentSetup:
    model_path: PurePosixPath
    group_size: int
    physical_rollout_batch_size: int
    rollout_max_new_tokens: int
    rollout_temperature: float
    rollout_top_p: float
    rollout_seed: int
    prompt: str
    prompt_sha256: str


def load_experiment_setup(profile_path: Path, prompt_path: Path) -> ExperimentSetup:
    """Read the configured G and require exactly one non-empty proxy prompt."""

    model_path = model_path_from_profile(profile_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    training = profile.get("training")
    if not isinstance(training, dict):
        raise ValueError("Profile must contain training settings")
    group_size = training.get("logical_group_size")
    if type(group_size) is not int or group_size < 2:
        raise ValueError("training.logical_group_size (G) must be an integer >= 2")
    physical_batch_size = training.get("physical_rollout_batch_size")
    if (
        type(physical_batch_size) is not int
        or physical_batch_size < 0
        or physical_batch_size > group_size
    ):
        raise ValueError("physical_rollout_batch_size must be 0 (auto) or between 1 and G")
    max_new_tokens = training.get("rollout_max_new_tokens")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("training.rollout_max_new_tokens must be a positive integer")
    temperature = training.get("rollout_temperature")
    if type(temperature) not in (int, float) or not 0 < temperature <= 2:
        raise ValueError("training.rollout_temperature must be in (0, 2]")
    top_p = training.get("rollout_top_p")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("training.rollout_top_p must be in (0, 1]")
    seed = training.get("rollout_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("training.rollout_seed must be a nonnegative integer")

    prompt_bytes = prompt_path.read_bytes()
    rows = prompt_rows(prompt_bytes.decode("utf-8").splitlines(keepends=True))
    if len(rows) != 1:
        raise ValueError("Proxy experiment requires exactly one non-empty prompt")
    return ExperimentSetup(
        model_path=model_path,
        group_size=group_size,
        physical_rollout_batch_size=physical_batch_size,
        rollout_max_new_tokens=max_new_tokens,
        rollout_temperature=float(temperature),
        rollout_top_p=float(top_p),
        rollout_seed=seed,
        prompt=rows[0][1],
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/models/qwen3_5_9b.yaml"))
    parser.add_argument(
        "--prompts", type=Path, default=Path("data/prompts/fictional_news_train_proxy_zh.txt")
    )
    args = parser.parse_args()
    setup = load_experiment_setup(args.profile, args.prompts)
    print(f"model={setup.model_path}")
    print(f"G={setup.group_size}")
    batch_size = setup.physical_rollout_batch_size or "auto (pending GPU probe)"
    print(f"physical_rollout_batch_size={batch_size}")
    print(f"rollout_max_new_tokens={setup.rollout_max_new_tokens}")
    print(f"rollout_temperature={setup.rollout_temperature}")
    print(f"rollout_top_p={setup.rollout_top_p}")
    print(f"rollout_seed={setup.rollout_seed}")
    print("prompt_count=1")
    print(f"prompt_sha256={setup.prompt_sha256}")
    print("preflight_only=true (no model load, Judge call, or training)")


if __name__ == "__main__":
    main()
