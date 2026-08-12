"""Content-derived chunks built from canonical parsed documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode()).hexdigest()}"


class ContentKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    OCR = "ocr"


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
        "chunk_id", "document_id", "text", "start", "end",
        "page_start", "page_end", "section", "kind",
    }

    @classmethod
    def create(
        cls, *, document_id: str, text: str, start: int, end: int,
        page_start: int, page_end: int, section: str | None = None,
        kind: ContentKind = ContentKind.TEXT,
    ) -> Chunk:
        payload = cls._identity_payload(document_id, text, start, end, kind)
        return cls(
            _stable_id("chk", payload), document_id, text, start, end,
            page_start, page_end, section, kind,
        )

    @staticmethod
    def _identity_payload(
        document_id: str, text: str, start: int, end: int, kind: ContentKind
    ) -> dict[str, object]:
        return {
            "document_id": document_id, "text": text, "start": start,
            "end": end, "kind": kind.value,
        }

    def __post_init__(self) -> None:
        if not self.document_id or not self.text.strip():
            raise ValueError("document_id and chunk text are required")
        values = (self.start, self.end, self.page_start, self.page_end)
        if any(type(value) is not int for value in values):
            raise ValueError("chunk offsets and pages must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid chunk offsets")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("invalid chunk page range")
        if self.section is not None and not self.section.strip():
            raise ValueError("section must be non-empty when provided")
        if not isinstance(self.kind, ContentKind):
            raise ValueError("kind must be a ContentKind")
        expected = _stable_id(
            "chk",
            self._identity_payload(
                self.document_id, self.text, self.start, self.end, self.kind
            ),
        )
        if self.chunk_id != expected:
            raise ValueError("chunk_id does not match canonical content")

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id, "document_id": self.document_id,
            "text": self.text, "start": self.start, "end": self.end,
            "page_start": self.page_start, "page_end": self.page_end,
            "section": self.section, "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Chunk:
        if not isinstance(value, Mapping) or set(value) != cls._FIELDS:
            raise ValueError("invalid chunk fields")
        try:
            kind = ContentKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid chunk kind") from exc
        return cls(
            value["chunk_id"], value["document_id"], value["text"], value["start"],
            value["end"], value["page_start"], value["page_end"], value["section"], kind,
        )  # type: ignore[arg-type]
