"""Explicit production dependency assembly with lazy heavyweight imports."""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.retrieval.bm25 import BM25Index
from aerollm.serving.adapters import ExistingMistralAdapter
from aerollm.serving.app import create_app
from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import RetrievedPassage, ServingDependencies
from aerollm.serving.fakes import IdentityPostprocessor

DEFAULT_PRODUCTION_CONFIG = Path("configs/serving/production_v1.toml")
CONFIG_ENVIRONMENT_VARIABLE = "AEROLLM_SERVING_CONFIG"


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    serving: ServingConfig
    corpus_path: Path
    model_path: Path
    model_id: str
    model_revision: str
    fp8_kernel_revision: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_id, "model_id"),
            (self.model_revision, "model_revision"),
            (self.fp8_kernel_revision, "fp8_kernel_revision"),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")

    @classmethod
    def from_toml(cls, path: Path) -> ProductionConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = _table(value, "runtime")
        model = _table(value, "model")
        if set(runtime) != {"corpus_path", "model_path"} or set(model) != {
            "id", "revision", "fp8_kernel_revision",
        }:
            raise ValueError("invalid production serving configuration fields")
        return cls(
            serving=ServingConfig.from_toml(path),
            corpus_path=Path(runtime["corpus_path"]),  # type: ignore[arg-type]
            model_path=Path(runtime["model_path"]),  # type: ignore[arg-type]
            model_id=model["id"],  # type: ignore[arg-type]
            model_revision=model["revision"],  # type: ignore[arg-type]
            fp8_kernel_revision=model["fp8_kernel_revision"],  # type: ignore[arg-type]
        )


class BM25CorpusRetriever:
    def __init__(self, corpus: CorpusManifest, index: BM25Index) -> None:
        self._chunks = {chunk.chunk_id: chunk.text for chunk in corpus.chunks}
        self._index = index

    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]:
        result = self._index.search(query, k=top_k)
        return tuple(
            RetrievedPassage(hit.chunk_id, self._chunks[hit.chunk_id], hit.score)
            for hit in result.hits
        )


def build_production_dependencies(config: ProductionConfig) -> ServingDependencies:
    """Load verified local artifacts; this is the only model-loading call path."""

    corpus_bytes = config.corpus_path.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode("utf-8"))
    index = BM25Index(corpus.chunks, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest())

    from aerollm.generation.ministral_backend import MinistralBackend

    model = MinistralBackend.from_local_path(
        config.model_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        fp8_kernel_revision=config.fp8_kernel_revision,
    )
    return ServingDependencies(
        backend=ExistingMistralAdapter(model),
        retriever=BM25CorpusRetriever(corpus, index),
        postprocessor=IdentityPostprocessor(),
    )


def create_production_app():  # type: ignore[no-untyped-def]
    """Uvicorn factory configured by ``AEROLLM_SERVING_CONFIG``."""

    config_path = Path(os.environ.get(CONFIG_ENVIRONMENT_VARIABLE, DEFAULT_PRODUCTION_CONFIG))
    production = ProductionConfig.from_toml(config_path)
    return create_app(
        config=production.serving,
        initializer=lambda: build_production_dependencies(production),
    )


def _table(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    table = value.get(name)
    if not isinstance(table, Mapping):
        raise ValueError(f"missing [{name}] production serving configuration")
    return table
