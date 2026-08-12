"""Normalize NTSB case records from verified immutable snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from aerollm.common.schemas import SourceDocument
from aerollm.data.sources import AviationReportRecord, ReportDocumentMetadata


class NTSBSnapshotSchemaError(ValueError):
    """The snapshot is valid JSON but not a supported NTSB response shape."""


def normalize_snapshot(document: SourceDocument) -> tuple[AviationReportRecord, ...]:
    """Verify and normalize a saved response; this function never calls the provider."""
    if document.source != "ntsb":
        raise ValueError(f"expected an ntsb snapshot, got {document.source!r}")
    body = _verified_body(document)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NTSBSnapshotSchemaError("NTSB snapshot is not valid UTF-8 JSON") from error

    cases = _case_items(payload)
    return tuple(_normalize_case(item, document) for item in cases)


def _verified_body(document: SourceDocument) -> bytes:
    path = Path(document.artifact_path)
    body = path.read_bytes()
    actual_digest = hashlib.sha256(body).hexdigest()
    if actual_digest != document.sha256:
        raise ValueError(
            f"snapshot digest mismatch for {path}: expected {document.sha256}, got {actual_digest}"
        )
    return body


def _case_items(payload: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        envelope = _casefolded(payload)
        found = next(
            (envelope[key] for key in ("cases", "data", "results", "value") if key in envelope),
            None,
        )
        if found is None:
            raise NTSBSnapshotSchemaError(
                "NTSB response must be a list or contain cases, data, results, or value"
            )
        items = found
    else:
        raise NTSBSnapshotSchemaError("NTSB response must be a JSON object or list")
    if not isinstance(items, list):
        raise NTSBSnapshotSchemaError("NTSB case collection must be a JSON list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise NTSBSnapshotSchemaError(f"NTSB case at index {index} must be an object")
        yield item


def _normalize_case(raw: Mapping[str, Any], document: SourceDocument) -> AviationReportRecord:
    values = _casefolded(raw)
    source_id = _required_text(values, "ntsbnumber", "accidentnumber", "casenumber")
    event_id = _optional_text(values, "eventid", "event_id")
    source_url = _optional_text(values, "caseurl", "sourceurl", "url") or document.source_url
    report_url = _optional_text(values, "reporturl", "finalreporturl")
    attributes = _attributes(values)
    return AviationReportRecord(
        source="ntsb",
        source_id=source_id,
        source_url=source_url,
        event_id=event_id,
        occurred_on=_optional_date(values, "eventdate", "occurreddate"),
        published_on=_optional_date(values, "publisheddate", "publicationdate"),
        attributes=attributes,
        documents=_documents(source_id, event_id, report_url),
    )


def _casefolded(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): item for key, item in value.items()}


def _required_text(values: Mapping[str, Any], *names: str) -> str:
    value = _optional_text(values, *names)
    if value is None:
        raise NTSBSnapshotSchemaError(f"NTSB case lacks required field {names[0]}")
    return value


def _optional_text(values: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = values.get(name.casefold())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _optional_date(values: Mapping[str, Any], *names: str) -> date | None:
    text = _optional_text(values, *names)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as error:
        raise NTSBSnapshotSchemaError(f"invalid {names[0]} date {text!r}") from error


def _attributes(values: Mapping[str, Any]) -> dict[str, str]:
    aliases = {
        "event_type": ("eventtype", "investigationtype"),
        "status": ("status", "investigationstatus"),
        "location": ("location", "eventlocation"),
        "country": ("eventcountry", "country"),
    }
    return {
        output_name: value
        for output_name, input_names in aliases.items()
        if (value := _optional_text(values, *input_names)) is not None
    }


def _documents(
    source_id: str, event_id: str | None, report_url: str | None
) -> tuple[ReportDocumentMetadata, ...]:
    if report_url is None:
        return ()
    return (
        ReportDocumentMetadata(
            source_id=f"{source_id}:report",
            source_url=report_url,
            kind="investigation-report",
            event_id=event_id,
        ),
    )
