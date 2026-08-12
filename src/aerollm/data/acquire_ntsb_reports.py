"""Bounded snapshot-first acquisition and parsing of NTSB report PDFs."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerollm.common.schemas import SourceDocument
from aerollm.data.manifests import ArtifactLink, DerivedArtifact, SourceManifest
from aerollm.data.normalize_ntsb import normalize_snapshot
from aerollm.data.ntsb import NTSBRequestError
from aerollm.data.ntsb_reports import NTSBReportSource
from aerollm.data.pdf import PDFParseError, parse_pdf_snapshot, write_parsed_document
from aerollm.data.snapshots import SnapshotStore, load_snapshot

MAX_REPORTS = 20
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snapshot and parse a bounded set of NTSB report PDFs"
    )
    parser.add_argument("--snapshot-metadata", type=Path, required=True)
    parser.add_argument("--max-reports", type=int, default=5)
    parser.add_argument("--config", type=Path, default=Path("configs/data/ntsb.toml"))
    parser.add_argument("--manifest", type=Path)
    return parser


def acquire_reports(
    case_snapshot: SourceDocument,
    *,
    config_path: Path,
    max_reports: int,
    manifest_path: Path | None = None,
    report_source: NTSBReportSource | None = None,
    created_at: datetime | None = None,
) -> SourceManifest:
    if not 1 <= max_reports <= MAX_REPORTS:
        raise ValueError(f"max_reports must be between 1 and {MAX_REPORTS}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    report_config: dict[str, Any] = config.get("reports", {})
    snapshot_config = config["snapshot"]
    max_bytes = int(report_config.get("max_bytes", 25 * 1024 * 1024))
    timeout_seconds = float(report_config.get("timeout_seconds", 30))
    if not 1 <= max_bytes <= MAX_PDF_BYTES:
        raise ValueError(f"configured max_bytes must be between 1 and {MAX_PDF_BYTES}")
    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"configured timeout_seconds must be greater than 0 and at most {MAX_TIMEOUT_SECONDS}"
        )
    available = tuple(
        document
        for case in normalize_snapshot(case_snapshot)
        for document in case.documents
        if document.media_type == "application/pdf"
    )
    if not available:
        raise ValueError("case snapshot contains no PDF report URLs")
    selected = available[:max_reports]

    source = report_source or NTSBReportSource(
        allowed_hosts=frozenset(
            report_config.get("allowed_hosts", ["www.ntsb.gov", "data.ntsb.gov"])
        ),
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
    )
    store = SnapshotStore(
        Path(snapshot_config["root"]),
        retained_headers=frozenset(snapshot_config.get("retain_response_headers", [])),
    )
    parsed_root = Path(report_config.get("parsed_root", "artifacts/parsed/ntsb"))
    report_snapshots: list[SourceDocument] = []
    derived: list[DerivedArtifact] = []
    links: list[ArtifactLink] = []
    for report in selected:
        response = source.fetch_document(report)
        report_snapshot = store.save(
            source=source.name,
            source_id=report.source_id,
            source_url=report.source_url,
            publisher="National Transportation Safety Board",
            response=response,
            event_id=report.event_id,
            usage_note=f"NTSB {report.kind}",
        )
        report_snapshots.append(report_snapshot)
        parsed = parse_pdf_snapshot(report_snapshot, max_bytes=source.max_bytes)
        parsed_path = parsed_root / report_snapshot.sha256[:2] / f"{report_snapshot.sha256}.json"
        parsed_digest = write_parsed_document(parsed, parsed_path)
        derived.append(
            DerivedArtifact(
                artifact_id=parsed.document_id,
                kind="parsed-document",
                sha256=parsed_digest,
                artifact_path=parsed_path.as_posix(),
            )
        )
        links.extend(
            (
                ArtifactLink(case_snapshot.sha256, report_snapshot.sha256, "has-report-document"),
                ArtifactLink(report_snapshot.sha256, parsed_digest, "parsed-as"),
            )
        )

    timestamp = created_at or datetime.now(UTC)
    manifest = SourceManifest(
        source="ntsb",
        created_at=timestamp,
        snapshots=(case_snapshot, *report_snapshots),
        derived_artifacts=tuple(derived),
        links=tuple(links),
    )
    output = manifest_path or Path("artifacts/manifests") / (
        f"ntsb-reports-{case_snapshot.sha256[:12]}.json"
    )
    manifest.write(output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        case_snapshot = load_snapshot(arguments.snapshot_metadata)
        manifest = acquire_reports(
            case_snapshot,
            config_path=arguments.config,
            max_reports=arguments.max_reports,
            manifest_path=arguments.manifest,
        )
    except (
        KeyError,
        NTSBRequestError,
        OSError,
        PDFParseError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
