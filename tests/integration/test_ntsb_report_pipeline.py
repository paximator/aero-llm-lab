import base64
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerollm.data.acquire_ntsb_reports import acquire_reports
from aerollm.data.ntsb_reports import NTSBReportSource
from aerollm.data.snapshots import SnapshotStore
from aerollm.data.sources import RemoteResponse

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ntsb"


def test_case_snapshot_to_pdf_snapshot_to_parsed_manifest(tmp_path) -> None:
    case_body = (FIXTURES / "cases-date-range-v2.json").read_bytes()
    case_snapshot = SnapshotStore(tmp_path / "snapshots").save(
        source="ntsb",
        source_id="cases-2026-01-01-2026-01-07",
        source_url="https://api.example.test/cases",
        publisher="NTSB",
        response=RemoteResponse(case_body, "application/json"),
        retrieved_at=datetime(2026, 1, 8, tzinfo=UTC),
    )
    pdf_body = base64.b64decode(
        (FIXTURES / "synthetic-report.pdf.base64").read_text(encoding="ascii")
    )

    def transport(*args):
        return RemoteResponse(pdf_body, "application/pdf", {"ETag": "synthetic-pdf"})

    report_source = NTSBReportSource(allowed_hosts=frozenset({"example.test"}), transport=transport)
    config = tmp_path / "ntsb.toml"
    config.write_text(
        f'''[snapshot]
root = "{(tmp_path / "snapshots").as_posix()}"
retain_response_headers = ["etag", "content-type"]
[reports]
parsed_root = "{(tmp_path / "parsed").as_posix()}"
''',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = acquire_reports(
        case_snapshot,
        config_path=config,
        max_reports=1,
        manifest_path=manifest_path,
        report_source=report_source,
        created_at=datetime(2026, 1, 9, tzinfo=UTC),
    )

    assert len(manifest.snapshots) == 2
    report_snapshot = manifest.snapshots[1]
    assert Path(report_snapshot.artifact_path).read_bytes() == pdf_body
    assert len(manifest.derived_artifacts) == 1
    parsed_path = Path(manifest.derived_artifacts[0].artifact_path)
    assert "Synthetic NTSB report page one." in parsed_path.read_text(encoding="utf-8")
    assert [link.relationship for link in manifest.links] == [
        "has-report-document",
        "parsed-as",
    ]
    assert manifest_path.exists()


def test_report_pipeline_rejects_unbounded_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        acquire_reports(None, config_path=tmp_path, max_reports=21)  # type: ignore[arg-type]
