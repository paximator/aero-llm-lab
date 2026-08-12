"""Measure local causal-model generation feasibility on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--fp8-kernel-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch
    from kernels import get_kernel
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    from transformers.integrations import finegrained_fp8

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this smoke benchmark")
    tokenizer = MistralCommonBackend.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False,
        torch_dtype="auto", device_map="cuda",
    ).eval()
    kernel = get_kernel(
        "kernels-community/finegrained-fp8",
        revision=args.fp8_kernel_revision,
        trust_remote_code=True,
    )
    finegrained_fp8._FINEGRAINED_FP8 = finegrained_fp8.FineGrainedFP8(
        matmul=kernel.matmul_2d,
        batched_matmul=kernel.matmul_batched,
        grouped_matmul=kernel.matmul_grouped,
    )
    load_seconds = time.perf_counter() - load_started
    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if input_ids.shape[-1] > args.max_input_tokens:
        raise ValueError("smoke prompt exceeds max input token budget")
    input_ids = input_ids.to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    completion = generated[0, input_ids.shape[-1]:]
    completion_tokens = int(completion.shape[-1])
    result = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "fp8_kernel_revision": args.fp8_kernel_revision,
        "model_config_sha256": _config_digest(args.model),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "prompt_tokens": int(input_ids.shape[-1]),
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "tokens_per_second": completion_tokens / generation_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        "completion": tokenizer.decode(completion, skip_special_tokens=True),
    }
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


def _config_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(path.glob("*.json")):
        digest.update(candidate.name.encode())
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
