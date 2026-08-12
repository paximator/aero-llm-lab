"""Content-addressed persistence for unmodified source responses."""

from __future__ import annotations

import hashlib
import json
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
            body_path.write_bytes(response.body)

        safe_headers = self._filtered_headers(response.headers)
        metadata = {
            "source": source,
            "source_id": source_id,
            "source_url": source_url,
            "retrieved_at": timestamp.astimezone(UTC).isoformat(),
            "sha256": digest,
            "content_type": response.content_type,
            "response_headers": safe_headers,
        }
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
            name.casefold(): value
            for name, value in headers.items()
            if name.casefold() in allowed
        }
