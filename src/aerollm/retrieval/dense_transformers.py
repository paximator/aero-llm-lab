"""Local-only Transformers encoder for E5-style dense retrieval."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class TransformersDenseEncoder:
    query_prefix = "query: "
    passage_prefix = "passage: "

    def __init__(
        self, model: Any, tokenizer: Any, torch: Any, *, model_id: str,
        model_revision: str, device: str, batch_size: int,
    ) -> None:
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self.model_id, self.model_revision = model_id, model_revision
        self.device, self.batch_size = device, batch_size
        self.dimension = int(model.config.hidden_size)

    @classmethod
    def from_local_path(
        cls, path: Path, *, model_id: str, model_revision: str,
        device: str = "cpu", batch_size: int = 32,
    ) -> TransformersDenseEncoder:
        if not path.is_dir() or batch_size < 1:
            raise ValueError("model path must exist and batch size must be positive")
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("run `uv sync --extra transformers`") from error
        options = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), **options)
        model = transformers.AutoModel.from_pretrained(str(path), **options).to(device).eval()
        return cls(
            model, tokenizer, torch, model_id=model_id, model_revision=model_revision,
            device=device, batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode([self.query_prefix + text for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._encode([self.passage_prefix + text for text in texts])

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        result: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(
                list(texts[start:start + self.batch_size]), max_length=512,
                padding=True, truncation=True, return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.extend(tuple(float(value) for value in row) for row in pooled.cpu().tolist())
        return tuple(result)
