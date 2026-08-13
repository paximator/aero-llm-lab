from datetime import date
from pathlib import Path

import pytest

from aerollm.common.schemas import Split
from aerollm.data.corpus_v2_selection import (
    SelectionConfig,
    build_replacement_review,
    build_review,
    freeze_reviewed_selection,
    select_candidates,
    select_replacement,
)
from aerollm.data.pilot import PilotCandidate


def _config(tmp_path: Path) -> SelectionConfig:
    return SelectionConfig(
        "selection", "seed", 4, tmp_path / "snapshots", tmp_path / "review.json",
        {Split.TRAIN: 2, Split.DEVELOPMENT: 1, Split.TEST: 1},
        ("draft", "approved", "needs_changes", "rejected"), "draft",
        date(2008, 1, 1), date(2017, 12, 31), 10, 31,
    )


def _candidate(name: str, year: int) -> PilotCandidate:
    return PilotCandidate(
        name, name, date(year, 1, 1), f"https://ntsb.test/{name}.pdf",
        "snapshot.json",
        {"severity": "Fatal" if year % 2 else "None", "weather": "VMC"},
    )


def test_selection_excludes_v1_and_assigns_exact_quotas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = tuple(_candidate(f"E{index}", 2018 + index) for index in range(6))

    selected = select_candidates(candidates, excluded_families={"E0"}, config=config)

    assert len(selected) == 4
    assert all(candidate.event_id != "E0" for candidate, _ in selected)
    assert sum(split is Split.TRAIN for _, split in selected) == 2
    assert sum(split is Split.DEVELOPMENT for _, split in selected) == 1
    assert sum(split is Split.TEST for _, split in selected) == 1


def test_selection_is_deterministic_and_review_starts_draft(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = tuple(_candidate(f"E{index}", 2018 + index) for index in range(5))

    first = select_candidates(candidates, excluded_families=set(), config=config)
    second = select_candidates(reversed(candidates), excluded_families=set(), config=config)
    review = build_review(
        first, config=config, excluded_families={"OLD"}, source_corpus_sha256="a" * 64,
    )

    assert first == second
    assert review["status"] == "review_required"
    assert all(item["review"]["status"] == "draft" for item in review["selections"])
    assert len(review["selection_sha256"]) == 64


def test_selection_refuses_to_weaken_approved_target(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="approved target"):
        select_candidates(
            (_candidate("E1", 2020),), excluded_families=set(), config=config,
        )


def test_replacement_preserves_split_and_excludes_reviewed_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    reviewed = build_review(
        select_candidates(
            tuple(_candidate(f"E{index}", 2018 + index) for index in range(4)),
            excluded_families={"OLD"}, config=config,
        ),
        config=config, excluded_families={"OLD"}, source_corpus_sha256="a" * 64,
    )
    rejected = next(item for item in reviewed["selections"] if item["proposed_split"] == "train")
    rejected["review"]["status"] = "rejected"

    candidate, split = select_replacement(
        (*(_candidate(f"E{index}", 2018 + index) for index in range(4)), _candidate("NEW", 2025)),
        reviewed=reviewed, config=config,
        rejected_event_id=rejected["event_family_id"],
    )
    follow_up = build_replacement_review(
        candidate, split, reviewed=reviewed, config=config,
        rejected_event_id=rejected["event_family_id"], verification_notes="public PDF verified",
    )

    assert candidate.event_id == "NEW"
    assert split is Split.TRAIN
    assert follow_up["replaces_event_family_id"] == rejected["event_family_id"]
    assert follow_up["selections"][0]["review"]["status"] == "draft"
    assert len(follow_up["replacement_sha256"]) == 64


def test_replacement_target_must_be_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    reviewed = build_review(
        select_candidates(
            tuple(_candidate(f"E{index}", 2018 + index) for index in range(4)),
            excluded_families=set(), config=config,
        ),
        config=config, excluded_families=set(), source_corpus_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="rejected event"):
        select_replacement(
            (_candidate("NEW", 2025),), reviewed=reviewed, config=config,
            rejected_event_id=reviewed["selections"][0]["event_family_id"],
        )


def test_freeze_combines_approved_replacement_and_preserves_quotas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    reviewed = build_review(
        select_candidates(
            tuple(_candidate(f"E{index}", 2018 + index) for index in range(4)),
            excluded_families={"OLD"}, config=config,
        ),
        config=config, excluded_families={"OLD"}, source_corpus_sha256="a" * 64,
    )
    for item in reviewed["selections"]:
        item["review"]["status"] = "approved"
    rejected = next(item for item in reviewed["selections"] if item["proposed_split"] == "train")
    rejected["review"]["status"] = "rejected"
    replacement = build_replacement_review(
        _candidate("NEW", 2025), Split.TRAIN, reviewed=reviewed, config=config,
        rejected_event_id=rejected["event_family_id"], verification_notes="verified",
    )
    replacement["selections"][0]["review"]["status"] = "approved"

    frozen = freeze_reviewed_selection(reviewed, replacement, config=config)

    assert frozen["status"] == "frozen"
    assert len(frozen["event_families"]) == 4
    expected = {"NEW"}
    expected.update(
        item["event_family_id"]
        for item in reviewed["selections"]
        if item is not rejected
    )
    assert {item["event_family_id"] for item in frozen["event_families"]} == expected
    assert all("review" not in item for item in frozen["event_families"])
    assert len(frozen["frozen_selection_sha256"]) == 64
