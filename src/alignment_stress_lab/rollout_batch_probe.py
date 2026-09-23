"""Benchmark 9B generation batch sizes on DGX; do not train or call a Judge."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

from alignment_stress_lab.experiment_setup import ExperimentSetup, load_experiment_setup
from alignment_stress_lab.rollout_batch import choose_rollout_batch_size


def _generate_tokens(model: Any, inputs: dict[str, Any], new_tokens: int, pad_token_id: int) -> int:
    import torch

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    return int(output.shape[-1] - inputs["input_ids"].shape[-1])


def probe_generation_batch(
    model: Any, tokenizer: Any, setup: ExperimentSetup, size: int
) -> float | None:
    """Return measured tokens/s, or None only for a CUDA out-of-memory failure."""

    import torch

    conversations = [[{"role": "user", "content": setup.prompt}] for _ in range(size)]
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    pad_token_id = tokenizer.pad_token_id
    print(f"Trying physical rollout batch size {size}... ", end="", flush=True)
    try:
        _generate_tokens(model, inputs, min(8, setup.rollout_max_new_tokens), pad_token_id)
        torch.cuda.synchronize()
        start = time.perf_counter()
        generated = _generate_tokens(model, inputs, setup.rollout_max_new_tokens, pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        throughput = size * generated / elapsed
        print(f"Ok ({throughput:.1f} tokens/s)", flush=True)
        return throughput
    except torch.cuda.OutOfMemoryError:
        print("Failed (CUDA out of memory)", flush=True)
        return None
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/models/qwen3_5_9b.yaml"))
    parser.add_argument(
        "--prompts", type=Path, default=Path("data/prompts/fictional_news_train_proxy_zh.txt")
    )
    parser.add_argument(
        "--force-auto",
        action="store_true",
        help="Benchmark auto mode even when the profile currently selects a manual size",
    )
    args = parser.parse_args()
    setup = load_experiment_setup(args.profile, args.prompts)
    if setup.physical_rollout_batch_size != 0 and not args.force_auto:
        print(f"Configured manual physical batch size: {setup.physical_rollout_batch_size}")
        print("No model loaded; use --force-auto to benchmark automatic selection.")
        return

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the generation batch probe")
    model_path = Path(str(setup.model_path))
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No local checkpoint at {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no usable pad token")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    if model.device.type != "cuda":
        raise RuntimeError(f"Model was not placed on CUDA: {model.device}")
    print(f"model={model_path} G={setup.group_size} max_new_tokens={setup.rollout_max_new_tokens}")
    selected = choose_rollout_batch_size(
        setup.group_size, lambda size: probe_generation_batch(model, tokenizer, setup, size)
    )
    print(f"Chosen generation batch size: {selected}")
    print("This is an inference-only result; training backward memory is untested.")


if __name__ == "__main__":
    main()
