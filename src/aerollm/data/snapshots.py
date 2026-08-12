"""Content-addressed persistence for unmodified source responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aerollm.common.schemas import SourceDocument
from aerollm.data.sources import RemoteResponse


@dataclass(frozen=True, slots=True)
class SnapshotStore:
    root: Path
    retained_headers: frozenset[str] = frozenset({"content-type", "etag", "last-modified"})

    def save(
        self,
        *,
        source: str,
        source_id: str,
        source_url: str,
        publisher: str,
        response: RemoteResponse,
        event_id: str | None = None,
        usage_note: str | None = None,
        request_parameters: Mapping[str, str] | None = None,
        retrieved_at: datetime | None = None,
    ) -> SourceDocument:
        timestamp = retrieved_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")

        digest = hashlib.sha256(response.body).hexdigest()
        directory = self.root / source / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        body_path = directory / f"{digest}.bin"
        metadata_path = directory / f"{digest}.json"

        # A digest collision can never silently replace different source bytes.
        if body_path.exists() and body_path.read_bytes() != response.body:
            raise RuntimeError(f"digest collision at {body_path}")
        if not body_path.exists():
            self._atomic_write(body_path, response.body)

        safe_headers = self._filtered_headers(response.headers)
        metadata = {
            "source": source,
            "source_id": source_id,
            "source_url": source_url,
            "retrieved_at": timestamp.astimezone(UTC).isoformat(),
            "sha256": digest,
            "content_type": response.content_type,
            "response_headers": safe_headers,
            "request_parameters": dict(sorted((request_parameters or {}).items())),
        }
        if not metadata_path.exists():
            self._atomic_write(
                metadata_path,
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
            )

        return SourceDocument(
            source=source,
            source_id=source_id,
            source_url=source_url,
            publisher=publisher,
            retrieved_at=timestamp,
            sha256=digest,
            artifact_path=body_path.as_posix(),
            event_id=event_id,
            usage_note=usage_note,
            metadata={"content_type": response.content_type, "response_headers": safe_headers},
        )

    def _filtered_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        allowed = {name.casefold() for name in self.retained_headers}
        return {
            name.casefold(): value for name, value in headers.items() if name.casefold() in allowed
        }

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """Publish complete files only; temporary files never escape the snapshot directory."""
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
