"""Local-only QLoRA memory-smoke and tiny-overfit runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QLoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    max_sequence_length: int
    learning_rate: float
    optimizer: str
    seed: int


def load_config(path: Path) -> QLoRAConfig:
    """Read and validate the stable subset used by the local runner."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    lora = raw["lora"]
    training = raw["training"]
    config = QLoRAConfig(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=tuple(str(item) for item in lora["target_modules"]),
        max_sequence_length=int(training["max_sequence_length"]),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        optimizer=str(training["optimizer"]),
        seed=int(training["seed"]),
    )
    if min(config.rank, config.alpha, config.max_sequence_length) < 1:
        raise ValueError("QLoRA integer settings must be positive")
    if not 0 <= config.dropout < 1 or config.learning_rate <= 0:
        raise ValueError("QLoRA dropout or learning rate is invalid")
    if not config.target_modules:
        raise ValueError("QLoRA target_modules cannot be empty")
    if config.optimizer != "paged_adamw_8bit":
        raise ValueError("only paged_adamw_8bit is supported")
    return config


def load_training_messages(path: Path, limit: int) -> list[list[dict[str, str]]]:
    """Load only the chat payload required for training."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    records = dataset.get("records")
    if not isinstance(records, list) or not 1 <= limit <= len(records):
        raise ValueError("training record limit is outside the dataset")
    messages = [record["messages"] for record in records[:limit]]
    if any(not isinstance(item, list) or len(item) != 3 for item in messages):
        raise ValueError("each training record must contain three chat messages")
    return messages


def tree_sha256(path: Path) -> str:
    """Fingerprint an adapter directory deterministically."""
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        if file.name == "run_manifest.json":
            continue
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Fingerprint one immutable run input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_training_chat(messages: list[dict[str, str]]) -> str:
    """Render a stable training format because the Base tokenizer has no chat template."""
    roles = ("system", "user", "assistant")
    if tuple(message.get("role") for message in messages) != roles:
        raise ValueError("training messages must be ordered system, user, assistant")
    return "\n\n".join(
        f"[{role.upper()}]\n{message['content'].strip()}"
        for role, message in zip(roles, messages, strict=True)
    )


def render_training_prompt(messages: list[dict[str, str]]) -> str:
    """Render the prefix masked from the causal language-model loss."""
    return render_training_chat(messages[:-1] + [{"role": "assistant", "content": ""}])


def training_order(record_count: int, steps: int, seed: int) -> list[int]:
    """Shuffle each epoch deterministically to avoid task-order recency collapse."""
    if record_count < 1 or steps < 1:
        raise ValueError("training order dimensions must be positive")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(record_count))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train, save, reload, and return an auditable run manifest."""
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA")
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise ValueError("--model must be an existing local directory")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("--output must not already exist")
    config = load_config(args.config)
    messages = load_training_messages(args.dataset, args.records)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_options = {
        "local_files_only": True,
        "device_map": {"": 0},
        "dtype": torch.bfloat16,
    }
    load_options["quantization_config"] = quantization
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **load_options,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    losses: list[float] = []
    order = training_order(len(messages), args.steps, config.seed)
    for step, record_index in enumerate(order):
        item = messages[record_index]
        text = render_training_chat(item)
        prompt = render_training_prompt(item)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_sequence_length,
        )
        encoded = {name: value.to("cuda") for name, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        prompt_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_sequence_length,
        )["input_ids"].shape[-1]
        labels[:, :prompt_tokens] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(**encoded, labels=labels).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)

    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    del optimizer, model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **load_options,
    )
    reloaded = PeftModel.from_pretrained(base, output, is_trainable=False)
    reloaded.eval()
    manifest = {
        "status": "passed",
        "gate": (
            "qlora_memory_smoke"
            if args.steps == 1
            else "qlora_tiny_overfit"
            if args.records <= 8
            else "qlora_first_sft_run"
        ),
        "quantization": "nf4",
        "model_path": str(model_path),
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "model_revision": tomllib.loads(args.config.read_text(encoding="utf-8"))["model"][
            "revision"
        ],
        "config": asdict(config),
        "records": args.records,
        "steps": args.steps,
        "losses": losses,
        "sampling": "deterministic_epoch_shuffle_v1",
        "loss_decreased": losses[-1] < losses[0] if len(losses) > 1 else None,
        "adapter_sha256": tree_sha256(output),
        "adapter_reload_passed": isinstance(reloaded, PeftModel),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/local_8gb_qlora.toml")
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/training/sft_v1_50.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
