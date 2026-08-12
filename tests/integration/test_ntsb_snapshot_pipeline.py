import json
from datetime import UTC, datetime
from pathlib import Path

from aerollm.data.manifests import SourceManifest
from aerollm.data.normalize_ntsb import normalize_snapshot
from aerollm.data.snapshots import SnapshotStore
from aerollm.data.sources import RemoteResponse

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ntsb" / "cases-date-range-v2.json"


def test_saved_snapshot_normalizes_to_neutral_records_and_manifest(tmp_path) -> None:
    raw_body = FIXTURE.read_bytes()
    snapshot = SnapshotStore(tmp_path / "snapshots").save(
        source="ntsb",
        source_id="cases-2026-01-01-2026-01-07",
        source_url=(
            "https://api.example.test/Common/v2/GetCasesByDateRange"
            "?startDate=2026-01-01&endDate=2026-01-07"
        ),
        publisher="National Transportation Safety Board",
        response=RemoteResponse(raw_body, "application/json", {"ETag": "synthetic-v1"}),
        request_parameters={"startDate": "2026-01-01", "endDate": "2026-01-07"},
        retrieved_at=datetime(2026, 1, 8, tzinfo=UTC),
    )

    records = normalize_snapshot(snapshot)
    assert [record.source_id for record in records] == ["CEN26FA001", "ERA26LA002"]
    assert records[0].occurred_on.isoformat() == "2026-01-02"
    assert records[0].attributes == {"event_type": "Accident", "status": "Completed"}
    assert records[0].documents[0].source_url == "https://example.test/reports/CEN26FA001"
    assert "providerOnlyField" not in records[0].attributes
    assert records[0].source_url == snapshot.source_url
    assert records[1].source_url == snapshot.source_url
    assert records[1].documents == ()
    assert records[1].attributes == {
        "country": "United States",
        "event_type": "Incident",
    }

    manifest_path = tmp_path / "manifests" / "ntsb-synthetic.json"
    SourceManifest(
        source="ntsb",
        created_at=datetime(2026, 1, 8, tzinfo=UTC),
        snapshots=(snapshot,),
    ).write(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["snapshots"][0]["sha256"] == snapshot.sha256
    assert manifest["snapshots"][0]["artifact_path"] == snapshot.artifact_path
