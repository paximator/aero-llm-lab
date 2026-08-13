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


def select_replacement(
    candidates: Iterable[PilotCandidate], *, reviewed: Mapping[str, object],
    config: SelectionConfig, rejected_event_id: str,
    ineligible_event_ids: set[str] | None = None,
) -> tuple[PilotCandidate, Split]:
    """Select one deterministic, untouched replacement for a rejected assignment."""
    selections = reviewed.get("selections")
    excluded = reviewed.get("excluded_event_families")
    if not isinstance(selections, list) or not isinstance(excluded, list):
        raise ValueError("invalid reviewed selection packet")
    rejected = [
        item for item in selections
        if isinstance(item, dict) and item.get("event_family_id") == rejected_event_id
    ]
    if len(rejected) != 1 or rejected[0].get("review", {}).get("status") != "rejected":
        raise ValueError("replacement target must be exactly one rejected event")
    split = Split(str(rejected[0]["proposed_split"]))
    blocked = {str(item) for item in excluded}
    blocked.update(
        str(item["event_family_id"]) for item in selections if isinstance(item, dict)
    )
    blocked.update(ineligible_event_ids or set())
    eligible = [
        candidate for candidate in candidates
        if candidate.event_id not in blocked and candidate.report_url
    ]
    if not eligible:
        raise ValueError("no untouched replacement candidates remain")
    chosen = min(
        eligible,
        key=lambda item: (
            _stable_key(
                config.seed, f"replacement\0{split.value}\0{item.event_id}",
            ),
            item.event_id,
        ),
    )
    return chosen, split


def build_replacement_review(
    candidate: PilotCandidate, split: Split, *, reviewed: Mapping[str, object],
    config: SelectionConfig, rejected_event_id: str,
    verification_notes: str,
) -> dict[str, object]:
    """Build a one-event follow-up packet without rewriting the owner's review."""
    selection = {
        "event_family_id": candidate.event_id,
        "event_id": candidate.event_id,
        "source_id": candidate.source_id,
        "occurred_on": candidate.occurred_on.isoformat(),
        "proposed_split": split.value,
        "report_url": candidate.report_url,
        "metadata": dict(sorted(candidate.attributes.items())),
        "automated_verification": verification_notes,
        "review": {"status": config.initial_status, "notes": ""},
    }
    identity = {
        "selection_id": f"{config.selection_id}-replacement",
        "seed": config.seed,
        "parent_selection_sha256": str(reviewed["selection_sha256"]),
        "replaces_event_family_id": rejected_event_id,
        "assignment": {
            "event_family_id": candidate.event_id,
            "split": split.value,
        },
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "review_required",
        "replacement_sha256": digest,
        **identity,
        "allowed_review_statuses": list(config.statuses),
        "instructions": "Review only this replacement event, then approve or reject it.",
        "selections": [selection],
    }


def freeze_reviewed_selection(
    reviewed: Mapping[str, object], replacement: Mapping[str, object], *,
    config: SelectionConfig,
) -> dict[str, object]:
    """Validate two approved review packets and return the immutable final selection."""
    base_identity = {
        key: reviewed[key]
        for key in (
            "selection_id", "seed", "source_corpus_sha256",
            "excluded_event_families", "assignments",
        )
    }
    base_digest = hashlib.sha256(
        json.dumps(base_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if reviewed.get("selection_sha256") != base_digest:
        raise ValueError("original selection digest does not match its assignments")
    if replacement.get("parent_selection_sha256") != base_digest:
        raise ValueError("replacement does not reference the reviewed selection")

    replacement_identity = {
        key: replacement[key]
        for key in (
            "selection_id", "seed", "parent_selection_sha256",
            "replaces_event_family_id", "assignment",
        )
    }
    replacement_digest = hashlib.sha256(
        json.dumps(replacement_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if replacement.get("replacement_sha256") != replacement_digest:
        raise ValueError("replacement digest does not match its assignment")

    base_items = reviewed.get("selections")
    replacement_items = replacement.get("selections")
    if not isinstance(base_items, list) or not isinstance(replacement_items, list):
        raise ValueError("review packets must contain selections")
    rejected = [
        item for item in base_items
        if isinstance(item, dict) and item.get("review", {}).get("status") == "rejected"
    ]
    approved = [
        item for item in base_items
        if isinstance(item, dict) and item.get("review", {}).get("status") == "approved"
    ]
    if len(rejected) != 1 or len(approved) != config.target - 1:
        raise ValueError("original review must contain 15 approvals and one rejection")
    rejected_id = rejected[0].get("event_family_id")
    if replacement.get("replaces_event_family_id") != rejected_id:
        raise ValueError("replacement target does not match the rejected event")
    if len(replacement_items) != 1:
        raise ValueError("replacement review must contain exactly one selection")
    replacement_item = replacement_items[0]
    if (
        not isinstance(replacement_item, dict)
        or replacement_item.get("review", {}).get("status") != "approved"
    ):
        raise ValueError("replacement must be approved before freeze")
    if replacement_item.get("proposed_split") != rejected[0].get("proposed_split"):
        raise ValueError("replacement must preserve the rejected event's split")

    final_items = [
        {key: value for key, value in item.items() if key != "review"}
        for item in (*approved, replacement_item)
    ]
    final_items.sort(key=lambda item: (str(item["proposed_split"]), str(item["event_family_id"])))
    family_ids = [str(item["event_family_id"]) for item in final_items]
    if len(family_ids) != config.target or len(set(family_ids)) != config.target:
        raise ValueError("frozen selection must contain unique event families")
    counts = {
        split: sum(item["proposed_split"] == split.value for item in final_items)
        for split in Split
    }
    if counts != dict(config.quotas):
        raise ValueError("frozen selection does not preserve split quotas")
    excluded = {str(item) for item in reviewed["excluded_event_families"]}
    if excluded.intersection(family_ids):
        raise ValueError("frozen selection overlaps the source corpus")

    frozen_identity = {
        "selection_id": config.selection_id,
        "seed": config.seed,
        "source_corpus_sha256": reviewed["source_corpus_sha256"],
        "event_families": final_items,
    }
    frozen_digest = hashlib.sha256(
        json.dumps(frozen_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "frozen",
        "frozen_selection_sha256": frozen_digest,
        "review_provenance": {
            "selection_sha256": base_digest,
            "replacement_sha256": replacement_digest,
        },
        **frozen_identity,
    }


def write_review(path: Path, review: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
