"""Local-only Qwen3.5-9B loading and generation smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import yaml


def model_path_from_profile(profile_path: Path) -> PurePosixPath:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if (
        not isinstance(profile, dict)
        or profile.get("profile_name") != "qwen3.5-9b"
        or profile.get("architecture") != "dense"
    ):
        raise ValueError("Expected the Dense Qwen3.5-9B profile")
    model_id = profile.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Set model_id to the local aligned checkpoint path")
    model_path = PurePosixPath(model_id)
    if not model_path.is_absolute():
        raise ValueError("model_id must be an absolute local path")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/models/qwen3_5_9b.yaml"),
    )
    args = parser.parse_args()
    model_path = Path(str(model_path_from_profile(args.profile)))
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No local checkpoint at {model_path}")

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    if model.device.type != "cuda":
        raise RuntimeError(f"Model was not placed on CUDA: {model.device}")

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with a short greeting."}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    completion = output[0, inputs["input_ids"].shape[-1] :]
    print(f"model={model_path}")
    print(f"device={model.device} torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"generated_tokens={completion.numel()}")
    print(f"response={tokenizer.decode(completion, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
