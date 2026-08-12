"""Canonical parsed documents and content-derived chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from aerollm.common.schemas import Split


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _keys(value: Mapping[str, object], expected: set[str], kind: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{kind} fields must be exactly {sorted(expected)}")


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode()).hexdigest()}"


class ContentKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    OCR = "ocr"


@dataclass(frozen=True, slots=True)
class PageSpan:
    page: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.page) is not int or self.page < 1:
            raise ValueError("page must be a positive integer")
        if type(self.start) is not int or type(self.end) is not int or self.start < 0:
            raise ValueError("page offsets must be non-negative integers")
        if self.end <= self.start:
            raise ValueError("page span end must be greater than start")

    def to_dict(self) -> dict[str, int]:
        return {"page": self.page, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PageSpan:
        if not isinstance(value, Mapping):
            raise ValueError("page span must be an object")
        _keys(value, {"page", "start", "end"}, "page span")
        return cls(value["page"], value["start"], value["end"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_document_id: str
    event_id: str
    event_family_id: str
    split: Split
    parser_version: str
    text: str
    pages: tuple[PageSpan, ...]

    _FIELDS: ClassVar[set[str]] = {
        "document_id",
        "source_document_id",
        "event_id",
        "event_family_id",
        "split",
        "parser_version",
        "text",
        "pages",
    }

    @classmethod
    def create(
        cls,
        *,
        source_document_id: str,
        event_id: str,
        event_family_id: str,
        split: Split,
        parser_version: str,
        text: str,
        pages: tuple[PageSpan, ...],
    ) -> Document:
        payload = cls._identity_payload(source_document_id, parser_version, text)
        return cls(
            _stable_id("doc", payload),
            source_document_id,
            event_id,
            event_family_id,
            split,
            parser_version,
            text,
            pages,
        )

    @staticmethod
    def _identity_payload(
        source_document_id: str, parser_version: str, text: str
    ) -> dict[str, str]:
        return {
            "source_document_id": source_document_id,
            "parser_version": parser_version,
            "text": text,
        }

    def __post_init__(self) -> None:
        for name in ("source_document_id", "event_id", "event_family_id", "parser_version", "text"):
            _text(getattr(self, name), name)
        if not isinstance(self.split, Split):
            raise ValueError("split must be a Split")
        if not isinstance(self.pages, tuple) or not self.pages:
            raise ValueError("pages must be a non-empty tuple")
        expected = _stable_id(
            "doc", self._identity_payload(self.source_document_id, self.parser_version, self.text)
        )
        if self.document_id != expected:
            raise ValueError("document_id does not match canonical content")
        cursor = 0
        for page in self.pages:
            if not isinstance(page, PageSpan) or page.start != cursor or page.end > len(self.text):
                raise ValueError("pages must be ordered, contiguous, and within document text")
            cursor = page.end
        if cursor != len(self.text):
            raise ValueError("pages must cover the complete document text")

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "source_document_id": self.source_document_id,
            "event_id": self.event_id,
            "event_family_id": self.event_family_id,
            "split": self.split.value,
            "parser_version": self.parser_version,
            "text": self.text,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Document:
        if not isinstance(value, Mapping):
            raise ValueError("document must be an object")
        _keys(value, cls._FIELDS, "document")
        pages = value["pages"]
        if not isinstance(pages, list):
            raise ValueError("pages must be a list")
        try:
            split = Split(value["split"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid document split") from exc
        return cls(
            value["document_id"],
            value["source_document_id"],
            value["event_id"],
            value["event_family_id"],
            split,
            value["parser_version"],
            value["text"],
            tuple(PageSpan.from_dict(page) for page in pages),  # type: ignore[arg-type]
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start: int
    end: int
    page_start: int
    page_end: int
    section: str | None
    kind: ContentKind

    _FIELDS: ClassVar[set[str]] = {
        "chunk_id",
        "document_id",
        "text",
        "start",
        "end",
        "page_start",
        "page_end",
        "section",
        "kind",
    }

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        text: str,
        start: int,
        end: int,
        page_start: int,
        page_end: int,
        section: str | None = None,
        kind: ContentKind = ContentKind.TEXT,
    ) -> Chunk:
        payload = cls._identity_payload(document_id, text, start, end, kind)
        return cls(
            _stable_id("chk", payload),
            document_id,
            text,
            start,
            end,
            page_start,
            page_end,
            section,
            kind,
        )

    @staticmethod
    def _identity_payload(
        document_id: str, text: str, start: int, end: int, kind: ContentKind
    ) -> dict[str, object]:
        return {
            "document_id": document_id,
            "text": text,
            "start": start,
            "end": end,
            "kind": kind.value,
        }

    def __post_init__(self) -> None:
        _text(self.document_id, "document_id")
        _text(self.text, "chunk text")
        if any(
            type(value) is not int
            for value in (self.start, self.end, self.page_start, self.page_end)
        ):
            raise ValueError("chunk offsets and pages must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid chunk offsets")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("invalid chunk page range")
        if self.section is not None:
            _text(self.section, "section")
        if not isinstance(self.kind, ContentKind):
            raise ValueError("kind must be a ContentKind")
        expected = _stable_id(
            "chk",
            self._identity_payload(self.document_id, self.text, self.start, self.end, self.kind),
        )
        if self.chunk_id != expected:
            raise ValueError("chunk_id does not match canonical content")

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Chunk:
        if not isinstance(value, Mapping):
            raise ValueError("chunk must be an object")
        _keys(value, cls._FIELDS, "chunk")
        try:
            kind = ContentKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid chunk kind") from exc
        return cls(
            value["chunk_id"],
            value["document_id"],
            value["text"],
            value["start"],
            value["end"],
            value["page_start"],
            value["page_end"],
            value["section"],
            kind,
        )  # type: ignore[arg-type]
