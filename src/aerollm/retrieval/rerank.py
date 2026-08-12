"""Cross-encoder reranking over a fixed first-stage candidate retriever."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from aerollm.common.documents import Chunk
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult


class CandidateRetriever(Protocol):
    def search(self, query: str, *, k: int = 10) -> RetrievalResult: ...


class PairScorer(Protocol):
    model_id: str
    model_revision: str

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class RerankManifest:
    index_id: str
    version: str
    candidate_index_id: str
    model_id: str
    model_revision: str
    candidate_k: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index_id": self.index_id, "version": self.version,
            "candidate_index_id": self.candidate_index_id,
            "model_id": self.model_id, "model_revision": self.model_revision,
            "candidate_k": self.candidate_k,
        }


class RerankedIndex:
    def __init__(
        self, candidate_retriever: CandidateRetriever, chunks: tuple[Chunk, ...],
        scorer: PairScorer, *, candidate_index_id: str, candidate_k: int,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self._candidate_retriever = candidate_retriever
        self._chunks: Mapping[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
        self._scorer = scorer
        identity = {
            "version": "cross-encoder-rerank-v1",
            "candidate_index_id": candidate_index_id,
            "model_id": scorer.model_id, "model_revision": scorer.model_revision,
            "candidate_k": candidate_k,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.manifest = RerankManifest(
            f"rerank_{digest}", "cross-encoder-rerank-v1", candidate_index_id,
            scorer.model_id, scorer.model_revision, candidate_k,
        )

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        started = time.perf_counter_ns()
        candidates = self._candidate_retriever.search(query, k=self.manifest.candidate_k)
        passages = [self._chunks[hit.chunk_id].text for hit in candidates.hits]
        scores = self._scorer.score(query, passages)
        if len(scores) != len(candidates.hits):
            raise ValueError("cross-encoder score count does not match candidates")
        ranked = sorted(
            zip(candidates.hits, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
        )[:k]
        minimum = min((score for _, score in ranked), default=0.0)
        hits = tuple(
            RetrievalHit(hit.chunk_id, rank, float(score - minimum))
            for rank, (hit, score) in enumerate(ranked, start=1)
        )
        return RetrievalResult(
            query, self.manifest.index_id, hits,
            float((time.perf_counter_ns() - started) / 1_000_000),
        )
