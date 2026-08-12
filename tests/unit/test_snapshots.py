import hashlib
import json
from datetime import UTC, datetime

from aerollm.data.snapshots import SnapshotStore, load_snapshot
from aerollm.data.sources import RemoteResponse


def test_snapshot_is_content_addressed_and_redacts_unlisted_headers(tmp_path) -> None:
    body = b'{"case": "CEN24LA001"}'
    store = SnapshotStore(tmp_path)
    record = store.save(
        source="ntsb",
        source_id="CEN24LA001",
        source_url="https://example.test/cases/CEN24LA001",
        publisher="NTSB",
        response=RemoteResponse(
            body=body,
            content_type="application/json",
            headers={"ETag": "abc", "Authorization": "secret", "X-Api-Key": "secret"},
        ),
        retrieved_at=datetime(2026, 8, 12, tzinfo=UTC),
        request_parameters={"startDate": "2026-08-01", "endDate": "2026-08-12"},
    )

    digest = hashlib.sha256(body).hexdigest()
    assert record.sha256 == digest
    assert (tmp_path / "ntsb" / digest[:2] / f"{digest}.bin").read_bytes() == body

    sidecar = json.loads(
        (tmp_path / "ntsb" / digest[:2] / f"{digest}.json").read_text(encoding="utf-8")
    )
    assert sidecar["response_headers"] == {"etag": "abc"}
    assert sidecar["request_parameters"]["startDate"] == "2026-08-01"
    assert "secret" not in json.dumps(sidecar)

    loaded = load_snapshot(tmp_path / "ntsb" / digest[:2] / f"{digest}.json")
    assert loaded == record


def test_saving_same_body_is_idempotent(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    arguments = {
        "source": "ntsb",
        "source_id": "case-1",
        "source_url": "https://example.test/case-1",
        "publisher": "NTSB",
        "response": RemoteResponse(body=b"same", content_type="application/pdf"),
        "retrieved_at": datetime(2026, 8, 12, tzinfo=UTC),
    }
    first = store.save(**arguments)
    second = store.save(**arguments)
    assert first.artifact_path == second.artifact_path
