"""Generate one fictional 9B proxy prompt group locally on DGX, without a Judge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from alignment_stress_lab.experiment_setup import load_experiment_setup
from alignment_stress_lab.rollout_batch import choose_rollout_batch_size
from alignment_stress_lab.rollout_batch_probe import probe_generation_batch


def completion_tokens(
    generated_ids: Sequence[int], eos_ids: set[int], max_new_tokens: int
) -> tuple[list[int], str]:
    """Keep the first EOS and exclude batch padding after it."""

    tokens = list(generated_ids[:max_new_tokens])
    for index, token_id in enumerate(tokens):
        if token_id in eos_ids:
            return tokens[: index + 1], "eos"
    if len(tokens) == max_new_tokens:
        return tokens, "length"
    return tokens, "invalid"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/models/qwen3_5_9b.yaml"))
    parser.add_argument(
        "--prompts", type=Path, default=Path("data/prompts/fictional_news_train_proxy_zh.txt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup = load_experiment_setup(args.profile, args.prompts)
    output_path = args.output
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Output or manifest already exists; choose a new --output")

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for proxy rollout generation")
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

    batch_size = setup.physical_rollout_batch_size
    if batch_size == 0:
        batch_size = choose_rollout_batch_size(
            setup.group_size, lambda size: probe_generation_batch(model, tokenizer, setup, size)
        )
    print(f"generation_batch_size={batch_size} G={setup.group_size}", flush=True)
    torch.manual_seed(setup.rollout_seed)
    torch.cuda.manual_seed_all(setup.rollout_seed)
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids = {tokenizer.eos_token_id}
    if not eos_ids:
        raise ValueError("No EOS token ID available for completion boundaries")

    records = []
    prompt_token_ids = None
    for start in range(0, setup.group_size, batch_size):
        size = min(batch_size, setup.group_size - start)
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
        if prompt_token_ids is None:
            prompt_token_ids = inputs["input_ids"][0].tolist()
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=setup.rollout_temperature,
                top_p=setup.rollout_top_p,
                max_new_tokens=setup.rollout_max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        prompt_width = inputs["input_ids"].shape[-1]
        for offset in range(size):
            token_ids, finish_reason = completion_tokens(
                generated[offset, prompt_width:].tolist(),
                eos_ids,
                setup.rollout_max_new_tokens,
            )
            records.append(
                {
                    "rollout_index": start + offset,
                    "prompt": setup.prompt,
                    "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                    "completion_token_ids": token_ids,
                    "generated_tokens": len(token_ids),
                    "finish_reason": finish_reason,
                }
            )
        print(f"generated {len(records)}/{setup.group_size} rollouts", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "profile_sha256": _sha256(args.profile),
        "prompts_sha256": setup.prompt_sha256,
        "output_sha256": _sha256(output_path),
        "group_size": setup.group_size,
        "physical_rollout_batch_size": batch_size,
        "prompt_token_ids": prompt_token_ids,
        "generation": {
            "thinking": False,
            "do_sample": True,
            "max_new_tokens": setup.rollout_max_new_tokens,
            "temperature": setup.rollout_temperature,
            "top_p": setup.rollout_top_p,
            "seed": setup.rollout_seed,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "judge": None,
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"wrote {len(records)} rollouts to {output_path}")


if __name__ == "__main__":
    main()
