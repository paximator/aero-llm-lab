"""Resumable construction of a diverse, bounded NTSB pilot corpus."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from aerollm.common.schemas import SourceDocument
from aerollm.data.chunking import ChunkingConfig, chunk_document
from aerollm.data.manifests import ArtifactLink, DerivedArtifact, SourceManifest
from aerollm.data.normalize_ntsb import normalize_snapshot
from aerollm.data.ntsb import NTSBSource
from aerollm.data.ntsb_reports import NTSBReportSource
from aerollm.data.pdf import parse_pdf_snapshot, write_parsed_document
from aerollm.data.snapshots import SnapshotStore, atomic_write, load_snapshot
from aerollm.data.sources import AviationReportRecord, ReportDocumentMetadata


@dataclass(frozen=True, slots=True)
class PilotConfig:
    seed: str
    target_reports: int
    max_discovery_requests: int
    discovery_window_days: int
    plan_path: Path
    report_manifest_root: Path
    chunk_root: Path
    failure_report: Path
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path) -> PilotConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version", "seed", "target_reports", "max_discovery_requests",
            "discovery_window_days", "plan_path", "report_manifest_root",
            "chunk_root", "failure_report",
        }
        if set(value) != expected:
            raise ValueError("invalid pilot configuration fields")
        return cls(
            seed=value["seed"], target_reports=value["target_reports"],
            max_discovery_requests=value["max_discovery_requests"],
            discovery_window_days=value["discovery_window_days"],
            plan_path=Path(value["plan_path"]),
            report_manifest_root=Path(value["report_manifest_root"]),
            chunk_root=Path(value["chunk_root"]),
            failure_report=Path(value["failure_report"]),
            schema_version=value["schema_version"],
        )

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.seed:
            raise ValueError("unsupported or incomplete pilot configuration")
        limits = (self.target_reports, self.max_discovery_requests, self.discovery_window_days)
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("pilot limits must be positive integers")
        if self.target_reports > 100 or self.max_discovery_requests > 100:
            raise ValueError("pilot target and discovery requests cannot exceed 100")
        if self.discovery_window_days > 31:
            raise ValueError("discovery windows cannot exceed 31 days")


@dataclass(frozen=True, slots=True)
class PilotCandidate:
    source_id: str
    event_id: str
    occurred_on: date
    report_url: str
    discovery_metadata: str
    attributes: Mapping[str, str]

    def features(self) -> tuple[str, ...]:
        values = [f"year:{self.occurred_on.year}"]
        values.extend(
            f"{key}:{self.attributes.get(key, 'unknown').casefold()}"
            for key in ("severity", "event_type", "weather", "report_type")
        )
        return tuple(values)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id, "event_id": self.event_id,
            "occurred_on": self.occurred_on.isoformat(), "report_url": self.report_url,
            "discovery_metadata": self.discovery_metadata,
            "attributes": dict(sorted(self.attributes.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PilotCandidate:
        expected = {
            "source_id", "event_id", "occurred_on", "report_url",
            "discovery_metadata", "attributes",
        }
        if set(value) != expected or not isinstance(value["attributes"], dict):
            raise ValueError("invalid pilot candidate fields")
        return cls(
            value["source_id"], value["event_id"], date.fromisoformat(value["occurred_on"]),
            value["report_url"], value["discovery_metadata"], value["attributes"],
        )


@dataclass(frozen=True, slots=True)
class PilotPlan:
    start: date
    end: date
    seed: str
    candidates: tuple[PilotCandidate, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "start": self.start.isoformat(),
            "end": self.end.isoformat(), "seed": self.seed,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def write(self, path: Path) -> None:
        content = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode()
        atomic_write(path, content)

    @classmethod
    def load(cls, path: Path) -> PilotPlan:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {"schema_version", "start", "end", "seed", "candidates"}
        if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
            raise ValueError("invalid pilot plan")
        return cls(
            date.fromisoformat(value["start"]), date.fromisoformat(value["end"]),
            value["seed"], tuple(PilotCandidate.from_dict(item) for item in value["candidates"]),
        )


def discovery_windows(
    start: date, end: date, *, requests: int, window_days: int
) -> tuple[tuple[date, date], ...]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    total_days = (end - start).days + 1
    count = min(requests, total_days)
    if count == 1:
        starts = [start]
    else:
        starts = [
            start + timedelta(days=index * (total_days - 1) // (count - 1))
            for index in range(count)
        ]
    return tuple((item, min(end, item + timedelta(days=window_days - 1))) for item in starts)


def select_diverse(
    candidates: Iterable[PilotCandidate], *, target: int, seed: str
) -> tuple[PilotCandidate, ...]:
    remaining = {candidate.source_id: candidate for candidate in candidates}
    selected: list[PilotCandidate] = []
    seen: set[str] = set()
    while remaining and len(selected) < target:
        ranked = sorted(
            remaining.values(),
            key=lambda item: (
                -sum(feature not in seen for feature in item.features()),
                _stable_key(seed, item.source_id), item.source_id,
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        seen.update(chosen.features())
        del remaining[chosen.source_id]
    return tuple(selected)


def discover_plan(
    start: date, end: date, *, pilot: PilotConfig, ntsb_config_path: Path
) -> PilotPlan:
    ntsb_config = tomllib.loads(ntsb_config_path.read_text(encoding="utf-8"))
    remote, snapshot = ntsb_config["remote"], ntsb_config["snapshot"]
    api_key = os.environ.get(remote["api_key_env"])
    if not api_key:
        raise ValueError(f"required environment variable {remote['api_key_env']} is not set")
    source = NTSBSource(
        base_url=remote["base_url"], api_key=api_key,
        endpoint_path=remote["endpoint_path"],
        timeout_seconds=float(remote.get("timeout_seconds", 30)),
    )
    store = SnapshotStore(
        Path(snapshot["root"]),
        retained_headers=frozenset(snapshot.get("retain_response_headers", [])),
    )
    candidates: list[PilotCandidate] = []
    for window_start, window_end in discovery_windows(
        start, end, requests=pilot.max_discovery_requests,
        window_days=pilot.discovery_window_days,
    ):
        metadata = next(source.list_reports(window_start, window_end))
        response = source.fetch_report(metadata)
        saved = store.save(
            source=source.name, source_id=metadata.source_id, source_url=metadata.source_url,
            publisher="National Transportation Safety Board", response=response,
            request_parameters=metadata.attributes,
            usage_note="Pilot-corpus NTSB discovery snapshot",
        )
        candidates.extend(_candidates_from_snapshot(saved))
    selected = select_diverse(candidates, target=pilot.target_reports, seed=pilot.seed)
    if len(selected) < pilot.target_reports:
        raise ValueError(
            f"only {len(selected)} cases with formal reports were found; "
            f"target is {pilot.target_reports}"
        )
    plan = PilotPlan(start, end, pilot.seed, selected)
    plan.write(pilot.plan_path)
    return plan


def materialize_plan(
    plan: PilotPlan,
    *,
    pilot: PilotConfig,
    ntsb_config_path: Path,
    chunk_config_path: Path,
    report_source: NTSBReportSource | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[Path, ...]:
    ntsb_config = tomllib.loads(ntsb_config_path.read_text(encoding="utf-8"))
    reports_config = ntsb_config["reports"]
    snapshot_config = ntsb_config["snapshot"]
    chunks_config = ChunkingConfig.from_dict(
        tomllib.loads(chunk_config_path.read_text(encoding="utf-8"))
    )
    source = report_source or NTSBReportSource(
        allowed_hosts=frozenset(reports_config["allowed_hosts"]),
        timeout_seconds=float(reports_config["timeout_seconds"]),
        max_bytes=int(reports_config["max_bytes"]),
    )
    store = SnapshotStore(
        Path(snapshot_config["root"]),
        retained_headers=frozenset(snapshot_config.get("retain_response_headers", [])),
    )
    parsed_root = Path(reports_config["parsed_root"])
    chunk_root = pilot.chunk_root
    completed: list[Path] = []
    failures: list[dict[str, str]] = []
    for candidate in plan.candidates:
        output = pilot.report_manifest_root / f"{candidate.source_id}.json"
        if output.exists():
            completed.append(output)
            continue
        try:
            discovery = load_snapshot(Path(candidate.discovery_metadata))
            report = ReportDocumentMetadata(
                source_id=f"{candidate.source_id}:report", source_url=candidate.report_url,
                kind="investigation-report", event_id=candidate.event_id,
            )
            response = source.fetch_document(report)
            timestamp = (now or (lambda: datetime.now(UTC)))()
            report_snapshot = store.save(
                source="ntsb", source_id=report.source_id, source_url=report.source_url,
                publisher="National Transportation Safety Board", response=response,
                event_id=candidate.event_id, usage_note="NTSB investigation-report",
                retrieved_at=timestamp,
            )
            document = parse_pdf_snapshot(report_snapshot, max_bytes=source.max_bytes)
            parsed_path = (
                parsed_root / report_snapshot.sha256[:2] / f"{report_snapshot.sha256}.json"
            )
            parsed_digest = write_parsed_document(document, parsed_path)
            chunk_manifest = chunk_document(document, chunks_config)
            chunk_manifest.write(chunk_root / f"{document.document_id}.json")
            manifest = SourceManifest(
                source="ntsb", created_at=timestamp,
                snapshots=(discovery, report_snapshot),
                derived_artifacts=(
                    DerivedArtifact(
                        document.document_id, "parsed-document", parsed_digest,
                        parsed_path.as_posix(),
                    ),
                ),
                links=(
                    ArtifactLink(discovery.sha256, report_snapshot.sha256, "has-report-document"),
                    ArtifactLink(report_snapshot.sha256, parsed_digest, "parsed-as"),
                ),
            )
            manifest.write(output)
            completed.append(output)
        except Exception as error:  # failures are isolated so the bounded batch can resume
            failures.append({"source_id": candidate.source_id, "error": str(error)})
    failure_content = {
        "schema_version": 1, "failed": failures,
        "completed_manifests": [path.as_posix() for path in completed],
    }
    atomic_write(
        pilot.failure_report,
        (json.dumps(failure_content, indent=2, sort_keys=True) + "\n").encode(),
    )
    return tuple(completed)


def _candidates_from_snapshot(snapshot: SourceDocument) -> list[PilotCandidate]:
    metadata_path = Path(snapshot.artifact_path).with_suffix(".json").as_posix()
    candidates: list[PilotCandidate] = []
    for record in normalize_snapshot(snapshot):
        document = next(iter(record.documents), None)
        if document is None or record.occurred_on is None:
            continue
        candidates.append(_candidate(record, document, metadata_path))
    return candidates


def _candidate(
    record: AviationReportRecord, document: ReportDocumentMetadata, metadata_path: str
) -> PilotCandidate:
    return PilotCandidate(
        record.source_id, record.event_id or record.source_id, record.occurred_on,
        document.source_url, metadata_path, record.attributes,
    )


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
