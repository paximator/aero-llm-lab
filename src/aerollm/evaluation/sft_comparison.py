"""Compare Base and SFT on reviewed development questions with fixed gold evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aerollm.evaluation.task_scoring import score_task_answer, validate_task_annotation
from aerollm.postprocessing import PostprocessorV1, RetrievedEvidence
from aerollm.training.qlora import render_training_prompt, tree_sha256

_SYSTEM = (
    "You answer aviation-report questions using only the supplied excerpt. "
    "Return the grounded-json-v1 JSON object and cite an exact span."
)


def build_messages(
    example: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[RetrievedEvidence]]:
    """Map one reviewed example to the exact inference contract used for SFT."""
    by_chunk: dict[str, list[str]] = {}
    for item in example["evidence"]:
        by_chunk.setdefault(str(item["chunk_id"]), []).append(str(item["quote"]))
    evidence = [
        RetrievedEvidence(chunk_id, "\n".join(quotes)) for chunk_id, quotes in by_chunk.items()
    ]
    chunks = "\n\n".join(
        f'<chunk id="{item.chunk_id}">\n{item.text}\n</chunk>' for item in evidence
    )
    if not chunks:
        chunks = "<no_evidence />"
    user = f"{example['question']}\n\n{chunks}"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ""},
    ]
    return messages, evidence


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate correctness and grounded-output behavior without hiding failures."""
    count = len(rows)
    return {
        "count": count,
        "task_accuracy_after_fail_closed": sum(bool(row["correct"]) for row in rows) / count,
        "valid_grounded_output_rate": sum(bool(row["valid_grounded_output"]) for row in rows)
        / count,
        "format_failure_rate": sum(bool(row["warnings"]) for row in rows) / count,
        "abstention_rate": sum(bool(row["abstained"]) for row in rows) / count,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows) / count,
    }


def _generate_variant(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    variant: str,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    processor = PostprocessorV1()
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        prepared = [build_messages(example) for example in batch]
        prompts = [render_training_prompt(messages) for messages, _ in prepared]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        for example, (_, evidence), sequence in zip(batch, prepared, generated, strict=True):
            raw = tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True).strip()
            processed = processor.process(raw, evidence)
            correct = score_task_answer(example, processed.answer)
            rows.append(
                {
                    "example_id": example["example_id"],
                    "task_type": example["task_type"],
                    "variant": variant,
                    "raw_output": raw,
                    "processed": processed.to_dict(),
                    "warnings": list(processed.warnings),
                    "correct": correct,
                    "valid_grounded_output": not processed.abstained and not processed.warnings,
                    "abstained": processed.abstained,
                    "latency_ms": elapsed_ms / len(batch),
                }
            )
        print(f"{variant}: completed {min(offset + len(batch), len(examples))}/{len(examples)}")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the pinned Base once, evaluate it, attach the adapter, and evaluate SFT."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    examples = dataset["examples"]
    for example in examples:
        validate_task_annotation(example)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    base_rows = _generate_variant(
        model,
        tokenizer,
        examples,
        "base_gold_context",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    sft_rows = _generate_variant(
        model,
        tokenizer,
        examples,
        "sft_gold_context",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    report = {
        "schema_version": 1,
        "status": "development_gold_context_comparison",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "adapter_sha256": tree_sha256(args.adapter),
        "decoding": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
        "aggregates": {
            "base_gold_context": aggregate(base_rows),
            "sft_gold_context": aggregate(sft_rows),
        },
        "predictions": [*base_rows, *sft_rows],
        "limitations": [
            "Gold evidence isolates generation quality and does not measure retrieval.",
            "Development annotations are reviewed, but this is not frozen test data.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/v2/development.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("generation limits must be positive")
    report = run(args)
    print(json.dumps(report["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
