"""Explicit production dependency assembly with lazy heavyweight imports."""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.retrieval.bm25 import BM25Index
from aerollm.retrieval.schemas import RetrievalResult
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
    retrieval_mode: str
    dense_index_path: Path
    dense_model_path: Path
    dense_model_id: str
    dense_model_revision: str
    dense_device: str
    dense_batch_size: int
    reranker_model_path: Path
    reranker_model_id: str
    reranker_model_revision: str
    reranker_device: str
    reranker_batch_size: int
    lexical_weight: float
    rrf_constant: int
    hybrid_candidate_k: int
    reranker_candidate_k: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_id, "model_id"),
            (self.model_revision, "model_revision"),
            (self.fp8_kernel_revision, "fp8_kernel_revision"),
            (self.dense_model_id, "dense_model_id"),
            (self.dense_model_revision, "dense_model_revision"),
            (self.dense_device, "dense_device"),
            (self.reranker_model_id, "reranker_model_id"),
            (self.reranker_model_revision, "reranker_model_revision"),
            (self.reranker_device, "reranker_device"),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        if self.retrieval_mode not in {"bm25", "hybrid_reranked"}:
            raise ValueError("retrieval mode must be bm25 or hybrid_reranked")
        if not 0.0 <= self.lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between zero and one")
        for value, name in (
            (self.dense_batch_size, "dense_batch_size"),
            (self.reranker_batch_size, "reranker_batch_size"),
            (self.rrf_constant, "rrf_constant"),
            (self.hybrid_candidate_k, "hybrid_candidate_k"),
            (self.reranker_candidate_k, "reranker_candidate_k"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_toml(cls, path: Path) -> ProductionConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = _table(value, "runtime")
        model = _table(value, "model")
        retrieval = _table(value, "retrieval")
        dense = _table(value, "dense")
        reranker = _table(value, "reranker")
        if set(runtime) != {
            "corpus_path", "model_path", "dense_index_path", "dense_model_path",
            "reranker_model_path",
        } or set(model) != {
            "id", "revision", "fp8_kernel_revision",
        }:
            raise ValueError("invalid production serving configuration fields")
        if set(retrieval) != {
            "mode", "lexical_weight", "rrf_constant", "hybrid_candidate_k",
            "reranker_candidate_k",
        } or set(dense) != {"id", "revision", "device", "batch_size"} or set(
            reranker
        ) != {"id", "revision", "device", "batch_size"}:
            raise ValueError("invalid production retrieval configuration fields")
        return cls(
            serving=ServingConfig.from_toml(path),
            corpus_path=Path(runtime["corpus_path"]),  # type: ignore[arg-type]
            model_path=Path(runtime["model_path"]),  # type: ignore[arg-type]
            model_id=model["id"],  # type: ignore[arg-type]
            model_revision=model["revision"],  # type: ignore[arg-type]
            fp8_kernel_revision=model["fp8_kernel_revision"],  # type: ignore[arg-type]
            retrieval_mode=retrieval["mode"],  # type: ignore[arg-type]
            dense_index_path=Path(runtime["dense_index_path"]),  # type: ignore[arg-type]
            dense_model_path=Path(runtime["dense_model_path"]),  # type: ignore[arg-type]
            dense_model_id=dense["id"],  # type: ignore[arg-type]
            dense_model_revision=dense["revision"],  # type: ignore[arg-type]
            dense_device=dense["device"],  # type: ignore[arg-type]
            dense_batch_size=dense["batch_size"],  # type: ignore[arg-type]
            reranker_model_path=Path(runtime["reranker_model_path"]),  # type: ignore[arg-type]
            reranker_model_id=reranker["id"],  # type: ignore[arg-type]
            reranker_model_revision=reranker["revision"],  # type: ignore[arg-type]
            reranker_device=reranker["device"],  # type: ignore[arg-type]
            reranker_batch_size=reranker["batch_size"],  # type: ignore[arg-type]
            lexical_weight=retrieval["lexical_weight"],  # type: ignore[arg-type]
            rrf_constant=retrieval["rrf_constant"],  # type: ignore[arg-type]
            hybrid_candidate_k=retrieval["hybrid_candidate_k"],  # type: ignore[arg-type]
            reranker_candidate_k=retrieval["reranker_candidate_k"],  # type: ignore[arg-type]
        )


class SearchIndex(Protocol):
    def search(self, query: str, *, k: int = 10) -> RetrievalResult: ...


class CorpusSearchRetriever:
    def __init__(self, corpus: CorpusManifest, index: SearchIndex) -> None:
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
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    retriever = build_production_retriever(config, corpus, corpus_sha256)

    from aerollm.generation.ministral_backend import MinistralBackend

    model = MinistralBackend.from_local_path(
        config.model_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        fp8_kernel_revision=config.fp8_kernel_revision,
    )
    return ServingDependencies(
        backend=ExistingMistralAdapter(model),
        retriever=retriever,
        postprocessor=IdentityPostprocessor(),
    )


def build_production_retriever(
    config: ProductionConfig, corpus: CorpusManifest, corpus_sha256: str,
) -> CorpusSearchRetriever:
    lexical = BM25Index(corpus.chunks, corpus_sha256=corpus_sha256)
    if config.retrieval_mode == "bm25":
        return CorpusSearchRetriever(corpus, lexical)

    from aerollm.retrieval.dense import DenseIndex
    from aerollm.retrieval.dense_transformers import TransformersDenseEncoder
    from aerollm.retrieval.hybrid import HybridIndex
    from aerollm.retrieval.rerank import RerankedIndex
    from aerollm.retrieval.rerank_transformers import TransformersCrossEncoder

    encoder = TransformersDenseEncoder.from_local_path(
        config.dense_model_path, model_id=config.dense_model_id,
        model_revision=config.dense_model_revision, device=config.dense_device,
        batch_size=config.dense_batch_size,
    )
    dense = DenseIndex.load(config.dense_index_path, encoder)
    expected_chunk_ids = tuple(sorted(chunk.chunk_id for chunk in corpus.chunks))
    if dense.manifest.corpus_sha256 != corpus_sha256:
        raise ValueError("dense index corpus digest does not match serving corpus")
    if dense.manifest.chunk_ids != expected_chunk_ids:
        raise ValueError("dense index chunks do not match serving corpus")
    hybrid = HybridIndex(
        lexical, dense, lexical_index_id=lexical.manifest.index_id,
        dense_index_id=dense.manifest.index_id, lexical_weight=config.lexical_weight,
        rrf_constant=config.rrf_constant, candidate_k=config.hybrid_candidate_k,
    )
    scorer = TransformersCrossEncoder.from_local_path(
        config.reranker_model_path, model_id=config.reranker_model_id,
        model_revision=config.reranker_model_revision, device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    )
    reranked = RerankedIndex(
        hybrid, corpus.chunks, scorer, candidate_index_id=hybrid.manifest.index_id,
        candidate_k=config.reranker_candidate_k,
    )
    return CorpusSearchRetriever(corpus, reranked)


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
