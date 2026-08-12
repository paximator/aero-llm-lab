"""Small, dependency-free BM25 lexical retrieval baseline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from aerollm.common.documents import Chunk
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult

_TOKEN = re.compile(r"[\w]+(?:[-./][\w]+)*", re.UNICODE)


def tokenize(value: str) -> tuple[str, ...]:
    """Case-fold a query or chunk into stable lexical tokens."""

    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(value))


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    tokenizer_version: str = "unicode-lexical-v1"

    def __post_init__(self) -> None:
        if type(self.k1) is not float or not math.isfinite(self.k1) or self.k1 <= 0.0:
            raise ValueError("k1 must be a positive float")
        if type(self.b) is not float or not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be a float between zero and one")
        if not isinstance(self.tokenizer_version, str) or not self.tokenizer_version.strip():
            raise ValueError("tokenizer_version is required")

    def to_dict(self) -> dict[str, object]:
        return {"k1": self.k1, "b": self.b, "tokenizer_version": self.tokenizer_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BM25Config:
        if not isinstance(value, Mapping) or set(value) != {"k1", "b", "tokenizer_version"}:
            raise ValueError("invalid BM25 config fields")
        return cls(value["k1"], value["b"], value["tokenizer_version"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IndexManifest:
    index_id: str
    version: str
    corpus_sha256: str
    chunk_ids: tuple[str, ...]
    config: BM25Config

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index_id, str)
            or not self.index_id.startswith("bm25_")
            or len(self.index_id) != 69
            or any(character not in "0123456789abcdef" for character in self.index_id[5:])
        ):
            raise ValueError("index_id must be a BM25 content identifier")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("index version is required")
        if (
            not isinstance(self.corpus_sha256, str)
            or len(self.corpus_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.corpus_sha256)
        ):
            raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.chunk_ids, tuple)
            or not self.chunk_ids
            or not all(isinstance(item, str) and item for item in self.chunk_ids)
            or len(set(self.chunk_ids)) != len(self.chunk_ids)
        ):
            raise ValueError("manifest chunk IDs must be non-empty and unique")
        if tuple(sorted(self.chunk_ids)) != self.chunk_ids:
            raise ValueError("manifest chunk IDs must be sorted")
        if not isinstance(self.config, BM25Config):
            raise ValueError("manifest config must be a BM25Config")
        identity = {
            "version": self.version,
            "corpus_sha256": self.corpus_sha256,
            "chunk_ids": list(self.chunk_ids),
            "config": self.config.to_dict(),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        expected = f"bm25_{hashlib.sha256(encoded).hexdigest()}"
        if self.index_id != expected:
            raise ValueError("index_id does not match manifest content")

    def to_dict(self) -> dict[str, object]:
        return {
            "index_id": self.index_id,
            "version": self.version,
            "corpus_sha256": self.corpus_sha256,
            "chunk_ids": list(self.chunk_ids),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> IndexManifest:
        fields = {"index_id", "version", "corpus_sha256", "chunk_ids", "config"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("invalid index manifest fields")
        chunk_ids, config = value["chunk_ids"], value["config"]
        if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
            raise ValueError("manifest chunk_ids must be a list of strings")
        if not isinstance(config, Mapping):
            raise ValueError("manifest config must be an object")
        return cls(
            value["index_id"],
            value["version"],
            value["corpus_sha256"],
            tuple(chunk_ids),
            BM25Config.from_dict(config),
        )  # type: ignore[arg-type]


class BM25Index:
    """An immutable in-memory index whose identity binds corpus and configuration."""

    def __init__(
        self,
        chunks: tuple[Chunk, ...],
        *,
        corpus_sha256: str,
        version: str = "bm25-v1",
        config: BM25Config | None = None,
    ) -> None:
        if not chunks or not all(isinstance(chunk, Chunk) for chunk in chunks):
            raise ValueError("BM25 requires a non-empty tuple of chunks")
        if len(corpus_sha256) != 64 or any(c not in "0123456789abcdef" for c in corpus_sha256):
            raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        self._chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        self._config = config or BM25Config()
        self._term_frequencies = tuple(Counter(tokenize(chunk.text)) for chunk in self._chunks)
        self._lengths = tuple(sum(frequencies.values()) for frequencies in self._term_frequencies)
        self._average_length = max(sum(self._lengths) / len(self._lengths), 1.0)
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        self._document_frequency = document_frequency
        identity = {
            "version": version,
            "corpus_sha256": corpus_sha256,
            "chunk_ids": [chunk.chunk_id for chunk in self._chunks],
            "config": self._config.to_dict(),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        index_id = f"bm25_{hashlib.sha256(encoded).hexdigest()}"
        self.manifest = IndexManifest(
            index_id, version, corpus_sha256, tuple(identity["chunk_ids"]), self._config
        )

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        started = time.perf_counter_ns()
        query_terms = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        count = len(self._chunks)
        for chunk, frequencies, length in zip(
            self._chunks, self._term_frequencies, self._lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self._config.k1 * (
                    1.0 - self._config.b + self._config.b * length / self._average_length
                )
                score += inverse_frequency * frequency * (self._config.k1 + 1.0) / denominator
            scored.append((score, chunk.chunk_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = tuple(
            RetrievalHit(chunk_id=chunk_id, rank=rank, score=float(score))
            for rank, (score, chunk_id) in enumerate(scored[:k], start=1)
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return RetrievalResult(query, self.manifest.index_id, hits, float(latency_ms))
