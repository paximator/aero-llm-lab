"""Persistent exact cosine retrieval over normalized dense embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aerollm.common.documents import Chunk
from aerollm.data.snapshots import atomic_write
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult


class DenseEncoder(Protocol):
    model_id: str
    model_revision: str
    dimension: int
    query_prefix: str
    passage_prefix: str

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...
    def encode_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    index_id: str
    version: str
    corpus_sha256: str
    model_id: str
    model_revision: str
    dimension: int
    query_prefix: str
    passage_prefix: str
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index_id": self.index_id, "version": self.version,
            "corpus_sha256": self.corpus_sha256, "model_id": self.model_id,
            "model_revision": self.model_revision, "dimension": self.dimension,
            "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
            "chunk_ids": list(self.chunk_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DenseIndexManifest:
        expected = {
            "index_id", "version", "corpus_sha256", "model_id", "model_revision",
            "dimension", "query_prefix", "passage_prefix", "chunk_ids",
        }
        if set(value) != expected or not isinstance(value["chunk_ids"], list):
            raise ValueError("invalid dense index manifest")
        return cls(
            str(value["index_id"]), str(value["version"]), str(value["corpus_sha256"]),
            str(value["model_id"]), str(value["model_revision"]), int(value["dimension"]),
            str(value["query_prefix"]), str(value["passage_prefix"]),
            tuple(str(item) for item in value["chunk_ids"]),
        )


class DenseIndex:
    def __init__(
        self, manifest: DenseIndexManifest, vectors: tuple[tuple[float, ...], ...],
        encoder: DenseEncoder,
    ) -> None:
        if len(vectors) != len(manifest.chunk_ids):
            raise ValueError("dense vector count does not match chunk IDs")
        if any(len(vector) != manifest.dimension for vector in vectors):
            raise ValueError("dense vector dimension mismatch")
        if (encoder.model_id, encoder.model_revision, encoder.dimension) != (
            manifest.model_id, manifest.model_revision, manifest.dimension,
        ):
            raise ValueError("encoder identity does not match dense index")
        self.manifest = manifest
        self._vectors = vectors
        self._encoder = encoder

    @classmethod
    def build(
        cls, chunks: tuple[Chunk, ...], *, corpus_sha256: str, encoder: DenseEncoder,
    ) -> DenseIndex:
        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        vectors = tuple(_normalize(vector) for vector in encoder.encode_passages(
            [chunk.text for chunk in ordered]
        ))
        identity = {
            "version": "dense-exact-v1", "corpus_sha256": corpus_sha256,
            "model_id": encoder.model_id, "model_revision": encoder.model_revision,
            "dimension": encoder.dimension, "query_prefix": encoder.query_prefix,
            "passage_prefix": encoder.passage_prefix,
            "chunk_ids": [chunk.chunk_id for chunk in ordered],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = DenseIndexManifest(
            f"dense_{digest}", "dense-exact-v1", corpus_sha256,
            encoder.model_id, encoder.model_revision, encoder.dimension,
            encoder.query_prefix, encoder.passage_prefix,
            tuple(chunk.chunk_id for chunk in ordered),
        )
        return cls(manifest, vectors, encoder)

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        if not query.strip() or k < 1:
            raise ValueError("query must be non-empty and k positive")
        started = time.perf_counter_ns()
        query_vector = _normalize(self._encoder.encode_queries([query])[0])
        scored = [
            (sum(a * b for a, b in zip(query_vector, vector, strict=True)), chunk_id)
            for chunk_id, vector in zip(self.manifest.chunk_ids, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = tuple(
            RetrievalHit(chunk_id, rank, float(max(score, 0.0)))
            for rank, (score, chunk_id) in enumerate(scored[:k], start=1)
        )
        return RetrievalResult(
            query, self.manifest.index_id, hits,
            float((time.perf_counter_ns() - started) / 1_000_000),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        values = array("f", (value for vector in self._vectors for value in vector))
        if sys.byteorder != "little":
            values.byteswap()
        atomic_write(directory / "vectors.f32", values.tobytes())
        atomic_write(
            directory / "manifest.json",
            (json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode(),
        )

    @classmethod
    def load(cls, directory: Path, encoder: DenseEncoder) -> DenseIndex:
        manifest = DenseIndexManifest.from_dict(
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        )
        values = array("f")
        values.frombytes((directory / "vectors.f32").read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        expected = len(manifest.chunk_ids) * manifest.dimension
        if len(values) != expected:
            raise ValueError("dense vector file size does not match manifest")
        vectors = tuple(
            tuple(values[start:start + manifest.dimension])
            for start in range(0, expected, manifest.dimension)
        )
        return cls(manifest, vectors, encoder)


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not values or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("encoder returned an invalid embedding")
    return tuple(value / norm for value in values)
