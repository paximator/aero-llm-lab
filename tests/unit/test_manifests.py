from datetime import UTC, datetime

import pytest

from aerollm.common.schemas import SourceDocument
from aerollm.data.manifests import SourceManifest


def _snapshot(source: str = "ntsb") -> SourceDocument:
    return SourceDocument(
        source=source,
        source_id="range-1",
        source_url="https://example.test/cases",
        publisher="NTSB",
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
        sha256="a" * 64,
        artifact_path="artifacts/snapshot.bin",
        metadata={"authorization": "must-not-leak"},
    )


def test_manifest_contains_only_reproducibility_fields(tmp_path) -> None:
    manifest = SourceManifest(
        source="ntsb",
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        snapshots=(_snapshot(),),
    )
    output = tmp_path / "manifest.json"
    manifest.write(output)
    text = output.read_text(encoding="utf-8")
    assert '"sha256": "aaaaaaaa' in text
    assert "must-not-leak" not in text


def test_manifest_rejects_mixed_sources() -> None:
    with pytest.raises(ValueError, match="manifest source"):
        SourceManifest(
            source="ntsb",
            created_at=datetime(2026, 1, 3, tzinfo=UTC),
            snapshots=(_snapshot("aaib"),),
        )
