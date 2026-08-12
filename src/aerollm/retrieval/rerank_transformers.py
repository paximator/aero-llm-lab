"""Local Transformers sequence-classification cross-encoder."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class TransformersCrossEncoder:
    def __init__(
        self, model: Any, tokenizer: Any, torch: Any, *, model_id: str,
        model_revision: str, device: str, batch_size: int,
    ) -> None:
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self.model_id, self.model_revision = model_id, model_revision
        self.device, self.batch_size = device, batch_size

    @classmethod
    def from_local_path(
        cls, path: Path, *, model_id: str, model_revision: str,
        device: str = "cpu", batch_size: int = 32,
    ) -> TransformersCrossEncoder:
        if not path.is_dir() or batch_size < 1:
            raise ValueError("model path must exist and batch size must be positive")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        options = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), **options)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(path), **options
        ).to(device).eval()
        return cls(
            model, tokenizer, torch, model_id=model_id, model_revision=model_revision,
            device=device, batch_size=batch_size,
        )

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        result: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch_passages = list(passages[start:start + self.batch_size])
            encoded = self._tokenizer(
                [query] * len(batch_passages), batch_passages, max_length=512,
                padding=True, truncation=True, return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits.squeeze(-1)
            result.extend(float(value) for value in logits.cpu().tolist())
        return tuple(result)
