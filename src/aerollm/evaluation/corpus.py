"""Cross-artifact validation for frozen grounded-QA datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, Split
from aerollm.evaluation.schemas import EvaluationDataset


@dataclass(frozen=True, slots=True)
class SourceManifestEntry:
    source_document_id: str
    event_id: str
    event_family_id: str
    split: Split
    sha256: str

    def __post_init__(self) -> None:
        for value in (self.source_document_id, self.event_id, self.event_family_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("source manifest identities must be non-empty strings")
        if not isinstance(self.split, Split):
            raise ValueError("source split must be a Split")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("source sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_document_id": self.source_document_id,
            "event_id": self.event_id,
            "event_family_id": self.event_family_id,
            "split": self.split.value,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceManifestEntry:
        fields = {"source_document_id", "event_id", "event_family_id", "split", "sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("invalid source manifest entry fields")
        try:
            split = Split(value["split"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid source manifest split") from exc
        return cls(
            value["source_document_id"],
            value["event_id"],
            value["event_family_id"],
            split,
            value["sha256"],
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    version: str
    sources: tuple[SourceManifestEntry, ...]
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]

    _FIELDS: ClassVar[set[str]] = {"version", "sources", "documents", "chunks"}

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("corpus version is required")
        if not all(
            isinstance(items, tuple) for items in (self.sources, self.documents, self.chunks)
        ):
            raise ValueError("corpus collections must be tuples")
        self._validate_integrity()

    def _validate_integrity(self) -> None:
        if not all(isinstance(item, SourceManifestEntry) for item in self.sources):
            raise ValueError("sources must contain SourceManifestEntry records")
        if not all(isinstance(item, Document) for item in self.documents):
            raise ValueError("documents must contain Document records")
        if not all(isinstance(item, Chunk) for item in self.chunks):
            raise ValueError("chunks must contain Chunk records")
        _unique(self.sources, "source_document_id", "source")
        sources_by_digest = _unique(self.sources, "sha256", "source digest")
        documents = _unique(self.documents, "document_id", "document")
        _unique(self.chunks, "chunk_id", "chunk")
        family_splits: dict[str, Split] = {}
        for source in self.sources:
            previous = family_splits.setdefault(source.event_family_id, source.split)
            if previous is not source.split:
                raise ValueError(f"event family {source.event_family_id!r} crosses source splits")
        for document in self.documents:
            source = sources_by_digest.get(document.source_sha256)
            if source is None:
                raise ValueError(
                    f"document references unknown source digest {document.source_sha256!r}"
                )
        for chunk in self.chunks:
            document = documents.get(chunk.document_id)
            if document is None:
                raise ValueError(f"chunk references unknown document {chunk.document_id!r}")
            if document.text[chunk.start : chunk.end] != chunk.text:
                raise ValueError(f"chunk {chunk.chunk_id!r} does not match document offsets")
            covered_pages = [
                page.page_number
                for page in document.pages
                if page.start_offset < chunk.end and page.end_offset > chunk.start
            ]
            if not covered_pages or (min(covered_pages), max(covered_pages)) != (
                chunk.page_start,
                chunk.page_end,
            ):
                raise ValueError(f"chunk {chunk.chunk_id!r} has an invalid page range")

    def validate_dataset(self, dataset: EvaluationDataset) -> None:
        sources = {item.source_document_id: item for item in self.sources}
        sources_by_digest = {item.sha256: item for item in self.sources}
        documents = {item.document_id: item for item in self.documents}
        chunks = {item.chunk_id: item for item in self.chunks}
        for example in dataset.examples:
            declared_sources = set(example.provenance.source_document_ids)
            if example.report_id not in declared_sources:
                raise ValueError(f"example {example.example_id!r} report is absent from provenance")
            for source_id in declared_sources:
                source = sources.get(source_id)
                if source is None:
                    raise ValueError(f"example references unknown source {source_id!r}")
                if (source.event_id, source.event_family_id, source.split) != (
                    example.event_id,
                    example.event_family_id,
                    example.split,
                ):
                    raise ValueError(
                        f"example {example.example_id!r} crosses event or split boundary"
                    )
            for evidence in example.evidence:
                chunk = chunks.get(evidence.chunk_id)
                if chunk is None:
                    raise ValueError(f"evidence references unknown chunk {evidence.chunk_id!r}")
                document = documents[chunk.document_id]
                source = sources_by_digest[document.source_sha256]
                if source.source_document_id not in declared_sources:
                    raise ValueError("evidence chunk is not from an example provenance source")
                if evidence.quote not in chunk.text:
                    raise ValueError(
                        f"evidence quote is not an exact span in {evidence.chunk_id!r}"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sources": [item.to_dict() for item in self.sources],
            "documents": [item.to_dict() for item in self.documents],
            "chunks": [item.to_dict() for item in self.chunks],
        }

    @classmethod
    def from_json(cls, value: str) -> CorpusManifest:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("corpus is not valid JSON") from exc
        if not isinstance(decoded, Mapping) or set(decoded) != cls._FIELDS:
            raise ValueError("invalid corpus fields")
        collections = (decoded["sources"], decoded["documents"], decoded["chunks"])
        if not all(isinstance(items, list) for items in collections):
            raise ValueError("corpus collections must be lists")
        return cls(
            decoded["version"],  # type: ignore[arg-type]
            tuple(SourceManifestEntry.from_dict(item) for item in decoded["sources"]),
            tuple(Document.from_dict(item) for item in decoded["documents"]),
            tuple(Chunk.from_dict(item) for item in decoded["chunks"]),
        )  # type: ignore[arg-type]


def _unique(records: tuple[object, ...], attribute: str, kind: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for record in records:
        key = getattr(record, attribute)
        if key in result:
            raise ValueError(f"duplicate {kind} ID: {key}")
        result[key] = record
    return result
