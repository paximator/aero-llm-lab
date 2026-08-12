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
class DerivedArtifact:
    artifact_id: str
    kind: str
    sha256: str
    artifact_path: str

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.kind or not self.artifact_path:
            raise ValueError("derived artifact identity, kind, and path are required")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("derived artifact sha256 must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ArtifactLink:
    parent_sha256: str
    child_sha256: str
    relationship: str

    def __post_init__(self) -> None:
        if not self.relationship:
            raise ValueError("artifact relationship is required")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source: str
    created_at: datetime
    snapshots: tuple[SourceDocument, ...]
    derived_artifacts: tuple[DerivedArtifact, ...] = ()
    links: tuple[ArtifactLink, ...] = ()
    schema_version: int = 2

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
        all_digests = set(digests) | {artifact.sha256 for artifact in self.derived_artifacts}
        if any(
            link.parent_sha256 not in all_digests or link.child_sha256 not in all_digests
            for link in self.links
        ):
            raise ValueError("manifest links must reference included artifacts")

    def to_dict(self) -> dict[str, object]:
        snapshots = sorted(self.snapshots, key=lambda item: (item.sha256, item.source_id))
        derived = sorted(self.derived_artifacts, key=lambda item: (item.sha256, item.artifact_id))
        links = sorted(
            self.links,
            key=lambda item: (item.parent_sha256, item.child_sha256, item.relationship),
        )
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
            "derived_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "sha256": artifact.sha256,
                    "artifact_path": artifact.artifact_path,
                }
                for artifact in derived
            ],
            "links": [
                {
                    "parent_sha256": link.parent_sha256,
                    "child_sha256": link.child_sha256,
                    "relationship": link.relationship,
                }
                for link in links
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
