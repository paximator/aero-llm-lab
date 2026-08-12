"""Deterministic page-aware chunking of canonical parsed documents."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerollm.common.documents import Chunk, ContentKind
from aerollm.common.schemas import Document, PageSpan
from aerollm.data.snapshots import atomic_write

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"(?m)^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Z0-9 /(),&'-]{4,80}$")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_characters: int = 1600
    overlap_characters: int = 256
    minimum_characters: int = 320
    boundary_search_characters: int = 400
    schema_version: int = 1

    def __post_init__(self) -> None:
        values = (
            self.target_characters,
            self.overlap_characters,
            self.minimum_characters,
            self.boundary_search_characters,
            self.schema_version,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("chunking configuration values must be positive integers")
        if self.overlap_characters >= self.target_characters:
            raise ValueError("overlap_characters must be smaller than target_characters")
        if self.minimum_characters > self.target_characters:
            raise ValueError("minimum_characters cannot exceed target_characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkingConfig:
        expected = {
            "schema_version", "target_characters", "overlap_characters",
            "minimum_characters", "boundary_search_characters",
        }
        if set(value) != expected:
            raise ValueError("invalid chunking configuration fields")
        return cls(**value)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, int]:
        return {
            "schema_version": self.schema_version,
            "target_characters": self.target_characters,
            "overlap_characters": self.overlap_characters,
            "minimum_characters": self.minimum_characters,
            "boundary_search_characters": self.boundary_search_characters,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ChunkManifest:
    document_id: str
    source_sha256: str
    config: ChunkingConfig
    chunks: tuple[Chunk, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported chunk manifest schema version")
        if not self.document_id or len(self.source_sha256) != 64:
            raise ValueError("manifest document identity is invalid")
        if not self.chunks:
            raise ValueError("chunk manifest cannot be empty")
        if any(chunk.document_id != self.document_id for chunk in self.chunks):
            raise ValueError("all chunks must belong to the manifest document")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkManifest:
        expected = {
            "schema_version", "document_id", "source_sha256", "config",
            "config_fingerprint", "chunks",
        }
        if set(value) != expected or not isinstance(value["chunks"], list):
            raise ValueError("invalid chunk manifest fields")
        config = ChunkingConfig.from_dict(value["config"])
        if value["config_fingerprint"] != config.fingerprint:
            raise ValueError("chunking configuration fingerprint mismatch")
        return cls(
            document_id=value["document_id"],
            source_sha256=value["source_sha256"],
            config=config,
            chunks=tuple(Chunk.from_dict(item) for item in value["chunks"]),
            schema_version=value["schema_version"],
        )

    def write(self, path: Path) -> str:
        content = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode()
        atomic_write(path, content)
        return hashlib.sha256(content).hexdigest()


def chunk_document(document: Document, config: ChunkingConfig) -> ChunkManifest:
    """Create stable chunks without changing or normalizing the source text."""
    chunks: list[Chunk] = []
    for page in document.pages:
        chunks.extend(_chunk_page(document, page, config))
    if not chunks:
        raise ValueError("document contains no non-empty page text")
    return ChunkManifest(document.document_id, document.source_sha256, config, tuple(chunks))


def _chunk_page(document: Document, page: PageSpan, config: ChunkingConfig) -> list[Chunk]:
    text = document.text
    cursor = page.start_offset
    page_end = page.end_offset
    result: list[Chunk] = []
    while cursor < page_end:
        start = _skip_whitespace(text, cursor, page_end)
        if start >= page_end:
            break
        header_boundary = _PARAGRAPH_BOUNDARY.search(text, start, min(start + 32, page_end))
        if (
            header_boundary is not None
            and _is_header_only_boilerplate(text[start : header_boundary.start()])
        ):
            cursor = header_boundary.end()
            continue
        desired = min(start + config.target_characters, page_end)
        end = _choose_end(text, start, desired, page_end, config)
        end = _trim_end(text, start, end)
        if end <= start:
            break
        chunk_text = text[start:end]
        if _is_header_only_boilerplate(chunk_text):
            cursor = end
            continue
        result.append(
            Chunk.create(
                document_id=document.document_id,
                text=chunk_text,
                start=start,
                end=end,
                page_start=page.page_number,
                page_end=page.page_number,
                section=_nearest_heading(text, page.start_offset, start, end),
                kind=_content_kind(chunk_text),
            )
        )
        if end >= page_end:
            break
        minimum_progress = min(end, start + config.minimum_characters)
        next_cursor = max(minimum_progress, end - config.overlap_characters)
        cursor = _next_boundary(text, next_cursor, end, page_end)
    return result


def _choose_end(
    text: str, start: int, desired: int, page_end: int, config: ChunkingConfig
) -> int:
    if desired >= page_end or page_end - start <= config.target_characters:
        return page_end
    lower = max(start + config.minimum_characters, desired - config.boundary_search_characters)
    upper = min(page_end, desired + config.boundary_search_characters)
    candidates = [match.end() for match in _PARAGRAPH_BOUNDARY.finditer(text, lower, upper)]
    if candidates:
        return min(candidates, key=lambda value: (abs(value - desired), value))
    sentence = max(text.rfind(". ", lower, desired), text.rfind(".\n", lower, desired))
    return sentence + 1 if sentence >= lower else desired


def _next_boundary(text: str, candidate: int, previous_end: int, page_end: int) -> int:
    paragraph = _PARAGRAPH_BOUNDARY.search(text, candidate, min(previous_end, page_end))
    return paragraph.end() if paragraph else candidate


def _skip_whitespace(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def _trim_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _nearest_heading(
    text: str, page_start: int, chunk_start: int, chunk_end: int
) -> str | None:
    headings = list(_HEADING.finditer(text, page_start, chunk_end))
    preceding = [heading for heading in headings if heading.start() <= chunk_start]
    selected = preceding[-1] if preceding else (headings[0] if headings else None)
    return selected.group(0).strip() if selected else None


def _content_kind(text: str) -> ContentKind:
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and sum("  " in line or "|" in line for line in lines) / len(lines) >= 0.4:
        return ContentKind.TABLE
    return ContentKind.TEXT


def _is_header_only_boilerplate(text: str) -> bool:
    return text.strip().casefold() == "ntsb"
