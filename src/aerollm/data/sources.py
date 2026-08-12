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


class ReportSource(Protocol):
    """The only boundary permitted to communicate with a source provider."""

    @property
    def name(self) -> str: ...

    def list_reports(self, start: date, end: date) -> Iterator[ReportMetadata]: ...

    def fetch_report(self, report: ReportMetadata) -> RemoteResponse: ...
