"""Versioned source manifests that bind dataset builds to immutable snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aerollm.common.schemas import SourceDocument


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source: str
    created_at: datetime
    snapshots: tuple[SourceDocument, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.source or not self.snapshots:
            raise ValueError("source and at least one snapshot are required")
        if any(snapshot.source != self.source for snapshot in self.snapshots):
            raise ValueError("all snapshots must belong to the manifest source")
        digests = [snapshot.sha256 for snapshot in self.snapshots]
        if len(digests) != len(set(digests)):
            raise ValueError("a manifest cannot contain duplicate snapshot digests")

    def to_dict(self) -> dict[str, object]:
        snapshots = sorted(self.snapshots, key=lambda item: (item.sha256, item.source_id))
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "snapshots": [
                {
                    "source_id": snapshot.source_id,
                    "source_url": snapshot.source_url,
                    "sha256": snapshot.sha256,
                    "artifact_path": snapshot.artifact_path,
                    "retrieved_at": snapshot.retrieved_at.astimezone(UTC).isoformat(),
                }
                for snapshot in snapshots
            ],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode()
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
