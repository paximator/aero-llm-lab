"""Typed retrieval records with strict validation and serialization."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        _required(self.chunk_id, "chunk_id")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if type(self.score) is not float or not math.isfinite(self.score) or self.score < 0.0:
            raise ValueError("score must be a non-negative float")

    def to_dict(self) -> dict[str, object]:
        return {"chunk_id": self.chunk_id, "rank": self.rank, "score": self.score}


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    index_id: str
    hits: tuple[RetrievalHit, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        _required(self.query, "query")
        _required(self.index_id, "index_id")
        if not isinstance(self.hits, tuple) or not all(
            isinstance(hit, RetrievalHit) for hit in self.hits
        ):
            raise ValueError("hits must be a tuple of RetrievalHit records")
        if [hit.rank for hit in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("hit ranks must be contiguous and start at one")
        if len({hit.chunk_id for hit in self.hits}) != len(self.hits):
            raise ValueError("retrieval hits must have unique chunk IDs")
        if (
            type(self.latency_ms) is not float
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0.0
        ):
            raise ValueError("latency_ms must be a non-negative float")

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "index_id": self.index_id,
            "hits": [hit.to_dict() for hit in self.hits],
            "latency_ms": self.latency_ms,
        }
