"""Deterministic metadata-only candidate selection for corpus v2."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aerollm.common.schemas import Split
from aerollm.data.ntsb import NTSBRequestError, NTSBSource
from aerollm.data.pilot import (
    PilotCandidate,
    _candidates_from_snapshot,
    discovery_windows,
    select_diverse,
)
from aerollm.data.snapshots import SnapshotStore, load_snapshot


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    selection_id: str
    seed: str
    target: int
    snapshot_root: Path
    review_output: Path
    quotas: Mapping[Split, int]
    statuses: tuple[str, ...]
    initial_status: str
    discovery_start: date
    discovery_end: date
    max_discovery_requests: int
    discovery_window_days: int

    @classmethod
    def load(cls, path: Path) -> SelectionConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        quotas = {Split(name): int(count) for name, count in value["split_quotas"].items()}
        review = value["review"]
        config = cls(
            str(value["selection_id"]), str(value["seed"]),
            int(value["target_event_families"]), Path(value["source_snapshot_root"]),
            Path(value["review_output"]), quotas,
            tuple(str(item) for item in review["allowed_statuses"]),
            str(review["initial_status"]),
            date.fromisoformat(value["discovery_start"]),
            date.fromisoformat(value["discovery_end"]),
            int(value["max_discovery_requests"]), int(value["discovery_window_days"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if set(self.quotas) != set(Split) or sum(self.quotas.values()) != self.target:
            raise ValueError("split quotas must define and sum to the selection target")
        if any(value < 1 for value in self.quotas.values()):
            raise ValueError("every split quota must be positive")
        if self.initial_status not in self.statuses or not self.seed or not self.selection_id:
            raise ValueError("invalid selection review configuration")
        if self.discovery_end < self.discovery_start:
            raise ValueError("invalid discovery date range")


def cached_candidates(root: Path) -> tuple[PilotCandidate, ...]:
    by_family: dict[str, PilotCandidate] = {}
    for metadata_path in sorted(root.glob("ntsb/*/*.json")):
        try:
            snapshot = load_snapshot(metadata_path)
            candidates = _candidates_from_snapshot(snapshot)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        for candidate in candidates:
            previous = by_family.get(candidate.event_id)
            if previous is None or candidate.source_id < previous.source_id:
                by_family[candidate.event_id] = candidate
    return tuple(by_family[key] for key in sorted(by_family))


def discover_candidates(
    config: SelectionConfig, ntsb_config_path: Path,
) -> tuple[dict[str, str], ...]:
    value = tomllib.loads(ntsb_config_path.read_text(encoding="utf-8"))
    remote, snapshot = value["remote"], value["snapshot"]
    api_key = os.environ.get(remote["api_key_env"])
    if not api_key:
        raise ValueError(f"required environment variable {remote['api_key_env']} is not set")
    source = NTSBSource(
        base_url=remote["base_url"], api_key=api_key,
        endpoint_path=remote["endpoint_path"],
        timeout_seconds=float(remote.get("timeout_seconds", 30)),
    )
    store = SnapshotStore(
        Path(snapshot["root"]),
        retained_headers=frozenset(snapshot.get("retain_response_headers", [])),
    )
    failures: list[dict[str, str]] = []
    for start, end in discovery_windows(
        config.discovery_start, config.discovery_end,
        requests=config.max_discovery_requests,
        window_days=config.discovery_window_days,
    ):
        try:
            metadata = next(source.list_reports(start, end, mode="Aviation"))
            response = source.fetch_report(metadata)
            store.save(
                source=source.name, source_id=metadata.source_id,
                source_url=metadata.source_url,
                publisher="National Transportation Safety Board", response=response,
                request_parameters=metadata.attributes,
                usage_note="Corpus-v2 metadata-only candidate discovery",
            )
        except (NTSBRequestError, OSError, StopIteration, ValueError) as error:
            failures.append(
                {"start": start.isoformat(), "end": end.isoformat(), "error": str(error)}
            )
    return tuple(failures)


def select_candidates(
    candidates: Iterable[PilotCandidate], *, excluded_families: set[str],
    config: SelectionConfig,
) -> tuple[tuple[PilotCandidate, Split], ...]:
    eligible = [
        candidate for candidate in candidates
        if candidate.event_id not in excluded_families and candidate.report_url
    ]
    selected = select_diverse(eligible, target=config.target, seed=config.seed)
    if len(selected) != config.target:
        raise ValueError(
            f"only {len(selected)} untouched event families with formal reports are cached; "
            f"the approved target is {config.target}"
        )
    ordered = sorted(
        selected,
        key=lambda item: (_stable_key(config.seed, item.event_id), item.event_id),
    )
    assignments: list[tuple[PilotCandidate, Split]] = []
    cursor = 0
    # Assign test first so its approved minimum cannot be eroded by later choices.
    for split in (Split.TEST, Split.DEVELOPMENT, Split.TRAIN):
        end = cursor + config.quotas[split]
        assignments.extend((candidate, split) for candidate in ordered[cursor:end])
        cursor = end
    return tuple(sorted(assignments, key=lambda item: (item[1].value, item[0].event_id)))


def build_review(
    assignments: tuple[tuple[PilotCandidate, Split], ...], *, config: SelectionConfig,
    excluded_families: set[str], source_corpus_sha256: str,
) -> dict[str, object]:
    selections = [
        {
            "event_family_id": candidate.event_id,
            "event_id": candidate.event_id,
            "source_id": candidate.source_id,
            "occurred_on": candidate.occurred_on.isoformat(),
            "proposed_split": split.value,
            "report_url": candidate.report_url,
            "metadata": dict(sorted(candidate.attributes.items())),
            "review": {"status": config.initial_status, "notes": ""},
        }
        for candidate, split in assignments
    ]
    identity = {
        "selection_id": config.selection_id,
        "seed": config.seed,
        "source_corpus_sha256": source_corpus_sha256,
        "excluded_event_families": sorted(excluded_families),
        "assignments": [
            {"event_family_id": item["event_family_id"], "split": item["proposed_split"]}
            for item in selections
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "review_required",
        "selection_sha256": digest,
        **identity,
        "allowed_review_statuses": list(config.statuses),
        "instructions": (
            "Review metadata only. Approve, request changes, or reject each event. "
            "Do not draft questions or inspect report contents before split approval."
        ),
        "selections": selections,
    }


def write_review(path: Path, review: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
