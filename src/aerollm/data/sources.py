"""Provider-neutral contracts for remote aviation report sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    source_id: str
    source_url: str
    event_id: str | None = None
    published_on: date | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RemoteResponse:
    body: bytes
    content_type: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReportDocumentMetadata:
    """Provider-neutral pointer to one published report document."""

    source_id: str
    source_url: str
    kind: str
    media_type: str = "application/pdf"
    event_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_url or not self.kind:
            raise ValueError("source_id, source_url, and kind are required")


@dataclass(frozen=True, slots=True)
class AviationReportRecord:
    """Stable normalized identity and dates shared by aviation providers."""

    source: str
    source_id: str
    source_url: str
    event_id: str | None = None
    occurred_on: date | None = None
    published_on: date | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    documents: tuple[ReportDocumentMetadata, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.source_id or not self.source_url:
            raise ValueError("source, source_id, and source_url are required")


class ReportSource(Protocol):
    """The only boundary permitted to communicate with a source provider."""

    @property
    def name(self) -> str: ...

    def list_reports(self, start: date, end: date) -> Iterator[ReportMetadata]: ...

    def fetch_report(self, report: ReportMetadata) -> RemoteResponse: ...
