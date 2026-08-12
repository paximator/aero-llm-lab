"""Reciprocal-rank fusion for lexical and dense retrieval results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult


class Retriever(Protocol):
    def search(self, query: str, *, k: int = 10) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class HybridManifest:
    index_id: str
    version: str
    lexical_index_id: str
    dense_index_id: str
    lexical_weight: float
    rrf_constant: int
    candidate_k: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index_id": self.index_id, "version": self.version,
            "lexical_index_id": self.lexical_index_id,
            "dense_index_id": self.dense_index_id,
            "lexical_weight": self.lexical_weight,
            "dense_weight": 1.0 - self.lexical_weight,
            "rrf_constant": self.rrf_constant, "candidate_k": self.candidate_k,
        }


class HybridIndex:
    def __init__(
        self, lexical: Retriever, dense: Retriever, *, lexical_index_id: str,
        dense_index_id: str, lexical_weight: float = 0.5, rrf_constant: int = 60,
        candidate_k: int = 100,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0 or rrf_constant < 1 or candidate_k < 1:
            raise ValueError("invalid hybrid retrieval configuration")
        self._lexical, self._dense = lexical, dense
        identity = {
            "version": "hybrid-rrf-v1", "lexical_index_id": lexical_index_id,
            "dense_index_id": dense_index_id, "lexical_weight": lexical_weight,
            "rrf_constant": rrf_constant, "candidate_k": candidate_k,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.manifest = HybridManifest(
            f"hybrid_{digest}", "hybrid-rrf-v1", lexical_index_id, dense_index_id,
            lexical_weight, rrf_constant, candidate_k,
        )

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        started = time.perf_counter_ns()
        lexical = self._lexical.search(query, k=self.manifest.candidate_k)
        dense = self._dense.search(query, k=self.manifest.candidate_k)
        scores: dict[str, float] = {}
        for result, weight in (
            (lexical, self.manifest.lexical_weight),
            (dense, 1.0 - self.manifest.lexical_weight),
        ):
            for hit in result.hits:
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (
                    self.manifest.rrf_constant + hit.rank
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        hits = tuple(
            RetrievalHit(chunk_id, rank, float(score))
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        )
        return RetrievalResult(
            query, self.manifest.index_id, hits,
            float((time.perf_counter_ns() - started) / 1_000_000),
        )
