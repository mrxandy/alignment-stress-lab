"""Generate a local Qwen3.5-9B baseline without calling an external Judge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from alignment_stress_lab.model_smoke import model_path_from_profile


def prompt_rows(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Keep source line numbers while ignoring blank lines."""

    return [
        (line_number, line.rstrip("\r\n"))
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/models/qwen3_5_9b.yaml")
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path("data/prompts/safety_baseline_dev_zh.txt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    model_path = Path(str(model_path_from_profile(args.profile)))
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No local checkpoint at {model_path}")
    with args.prompts.open(encoding="utf-8") as source:
        rows = prompt_rows(source)
    if not rows:
        raise ValueError("Prompt file contains no non-empty lines")
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = args.output
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Output or manifest already exists; choose a new --output")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for baseline generation")
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

    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for line_number, prompt in rows:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(model.device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            completion = generated[0, inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(completion, skip_special_tokens=True)
            record = {
                "line_number": line_number,
                "prompt": prompt,
                "response": response,
                "generated_tokens": completion.numel(),
                "finish_reason": (
                    "eos" if completion.numel() and completion[-1].item() in eos_ids else "length"
                ),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"completed line {line_number}", flush=True)

    manifest = {
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "profile_sha256": _sha256(args.profile),
        "prompts_sha256": _sha256(args.prompts),
        "prompt_count": len(rows),
        "generation": {
            "thinking": False,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "judge": None,
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"wrote {len(rows)} responses to {output_path}")


if __name__ == "__main__":
    main()
