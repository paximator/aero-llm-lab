import copy
import hashlib
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.training.sft_data import build_sft_dataset, validate_sft_dataset


def _corpus() -> CorpusManifest:
    records = []
    for index, split in enumerate((Split.TRAIN, Split.TRAIN, Split.TEST), start=1):
        text = (
            f"The accident airplane {index} landed on the runway after the flight crew "
            "reported an engine warning and followed the published airport procedure. "
            f"The NTSB investigation for event {index} documented aircraft damage and "
            "reviewed the pilot actions during landing."
        )
        digest = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        document = Document(f"doc-{digest}", digest, text, (PageSpan(1, 0, len(text)),), "fixture")
        chunk = Chunk.create(
            document_id=document.document_id,
            text=text,
            start=0,
            end=len(text),
            page_start=1,
            page_end=1,
        )
        source = SourceManifestEntry(
            f"report-{index}", f"event-{index}", f"family-{index}", split, digest
        )
        records.append((source, document, chunk))
    return CorpusManifest(
        "fixture-v1",
        tuple(record[0] for record in records),
        tuple(record[1] for record in records),
        tuple(record[2] for record in records),
    )


def test_builder_selects_only_train_families_and_exact_evidence() -> None:
    corpus = _corpus()

    dataset = build_sft_dataset(
        corpus,
        corpus_sha256="a" * 64,
        target_records=2,
        max_records_per_family=1,
    )

    assert len(dataset["records"]) == 2
    assert {record["event_family_id"] for record in dataset["records"]} == {
        "family-1",
        "family-2",
    }
    validate_sft_dataset(dataset, corpus)


def test_validator_rejects_test_report_leakage() -> None:
    corpus = _corpus()
    dataset = build_sft_dataset(
        corpus, corpus_sha256="a" * 64, target_records=1, max_records_per_family=1
    )
    leaked = copy.deepcopy(dataset)
    leaked["records"][0]["report_id"] = "report-3"

    with pytest.raises(ValueError, match="train-only"):
        validate_sft_dataset(leaked, corpus)


def test_validator_rejects_duplicate_messages() -> None:
    corpus = _corpus()
    dataset = build_sft_dataset(
        corpus,
        corpus_sha256="a" * 64,
        target_records=2,
        max_records_per_family=1,
    )
    duplicate = copy.deepcopy(dataset)
    duplicate_record_id = duplicate["records"][1]["record_id"]
    duplicate["records"][1] = copy.deepcopy(duplicate["records"][0])
    duplicate["records"][1]["record_id"] = duplicate_record_id

    with pytest.raises(ValueError, match="duplicate SFT messages"):
        validate_sft_dataset(duplicate, corpus)


def test_repository_dataset_has_expected_validation_stage_shape() -> None:
    import json

    root = Path(__file__).parents[2]
    dataset = json.loads((root / "data/training/sft_v1_50.json").read_text(encoding="utf-8"))

    assert len(dataset["records"]) == 50
    assert len({record["event_family_id"] for record in dataset["records"]}) == 13
    assert all(record["review_status"] == "sample_review_required" for record in dataset["records"])
    review = json.loads(
        (root / "data/training/sft_v1_50.sample_review.json").read_text(encoding="utf-8")
    )
    record_ids = {record["record_id"] for record in dataset["records"]}
    assert {item["record_id"] for item in review["reviews"]} <= record_ids
