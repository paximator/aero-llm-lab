import hashlib
from datetime import UTC, datetime

import pytest

from aerollm.common.schemas import SourceDocument
from aerollm.data.normalize_ntsb import NTSBSnapshotSchemaError, normalize_snapshot


def _document(path, body: bytes, *, digest: str | None = None) -> SourceDocument:
    path.write_bytes(body)
    return SourceDocument(
        source="ntsb",
        source_id="date-range",
        source_url="https://api.example.test/cases",
        publisher="NTSB",
        retrieved_at=datetime(2026, 1, 5, tzinfo=UTC),
        sha256=digest or hashlib.sha256(body).hexdigest(),
        artifact_path=path.as_posix(),
    )


def test_normalizer_rejects_changed_snapshot_bytes(tmp_path) -> None:
    document = _document(tmp_path / "snapshot.bin", b"[]", digest="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        normalize_snapshot(document)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "UTF-8 JSON"),
        (b'{"unexpected": []}', "contain cases"),
        (b'{"cases": {}}', "collection must be a JSON list"),
        (b'{"cases": [{}]}', "required field ntsbnumber"),
        (b'{"cases": [{"ntsbNumber": "case", "eventDate": "soon"}]}', "eventdate"),
    ],
)
def test_normalizer_reports_unsupported_schema(tmp_path, body, message) -> None:
    document = _document(tmp_path / "snapshot.bin", body)
    with pytest.raises(NTSBSnapshotSchemaError, match=message):
        normalize_snapshot(document)
