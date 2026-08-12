"""Stable records exchanged across AeroLLM subsystems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Split(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source: str
    source_id: str
    source_url: str
    publisher: str
    retrieved_at: datetime
    sha256: str
    artifact_path: str
    event_id: str | None = None
    published_at: datetime | None = None
    usage_note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")
        if not self.source or not self.source_id:
            raise ValueError("source and source_id are required")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieved_at"] = self.retrieved_at.astimezone(UTC).isoformat()
        if self.published_at is not None:
            value["published_at"] = self.published_at.isoformat()
        return value


@dataclass(frozen=True, slots=True)
class PageSpan:
    page_number: int
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.page_number < 1 or self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("invalid page span")


@dataclass(frozen=True, slots=True)
class Document:
    """Provider-neutral parsed document with page-addressable text."""

    document_id: str
    source_sha256: str
    text: str
    pages: tuple[PageSpan, ...]
    parser: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id or not self.parser:
            raise ValueError("document_id and parser are required")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a SHA-256 digest")
        if any(page.end_offset > len(self.text) for page in self.pages):
            raise ValueError("page span exceeds document text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    chunk_id: str
    quote: str

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.quote.strip():
            raise ValueError("evidence requires a chunk_id and non-empty quote")


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str | None
    citations: tuple[EvidenceSpan, ...]
    abstained: bool
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        if self.abstained:
            if self.answer is not None or self.citations:
                raise ValueError("an abstention cannot contain an answer or citations")
            if not self.abstention_reason:
                raise ValueError("an abstention requires a reason")
        elif not self.answer or not self.citations:
            raise ValueError("a grounded answer requires text and at least one citation")
