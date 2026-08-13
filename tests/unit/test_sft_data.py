import copy
import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.training.sft_data import (
    _quality_rejection_reason,
    build_sft_dataset,
    validate_sft_dataset,
)
from aerollm.training.sft_review_cli import main as review_main


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
    assert dataset["records"][0]["source_pages"] == [1]
    assert dataset["records"][0]["investigation_url"].endswith("event-1.aspx")
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

    with pytest.raises(ValueError, match="duplicate SFT answers"):
        validate_sft_dataset(duplicate, corpus)


def test_repository_dataset_has_expected_validation_stage_shape() -> None:
    root = Path(__file__).parents[2]
    dataset = json.loads((root / "data/training/sft_v1_50.json").read_text(encoding="utf-8"))

    assert len(dataset["records"]) == 50
    assert len({record["event_family_id"] for record in dataset["records"]}) == 13
    assert all(record["review_status"] == "sample_review_required" for record in dataset["records"])
    review = json.loads(
        (root / "data/training/sft_v1_50.sample_review.json").read_text(encoding="utf-8")
    )
    assert review["sample_size"] == 10
    assert [item["record_id"] for item in review["reviews"]] == [
        record["record_id"] for record in dataset["records"][:10]
    ]


def test_review_cli_shows_official_url_pages_and_expected_answer(capsys) -> None:  # type: ignore[no-untyped-def]
    root = Path(__file__).parents[2]
    dataset_path = root / "data/training/sft_v1_50.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    record_id = dataset["records"][0]["record_id"]

    result = review_main(
        [
            str(dataset_path),
            "--record",
            record_id,
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "https://www.ntsb.gov/investigations/Pages/ANC20MA010.aspx" in output
    assert "PDF page(s):" in output
    assert "Expected answer:" in output


@pytest.mark.parametrize(
    "bad_text",
    [
        "NTSB Aircraft Accident Report 1.2.2 Certificate History The pilot landed safely.",
        "Aviation Accident Report variable for the landing configuration was reverse power.",
        "The flight crew found that theneed for training was important after the accident.",
        "The operator stated that we a re concerned about the aircraft controls.",
        "11:17:04 RDO-1 okay could you have medical personnel meet us on the runway.",
        "The airplane touched down on 2 A circular bright spot can be seen on the flap.",
        "It was comprehensive. We reviewed the pilot history and administered a test.",
        "A-24-9 Require retrofit of cockpit voice recorders on transport airplanes.",
        "Departure from Controlled Flight, Trans-Pacific Air Charter, Learjet 35A, New Jersey.",
        "Workforce manufacturing 83 The FAA grounded the Boeing aircraft after the accident.",
        "Continue the takeoff normally when runway congestion prevents an engine run-up.",
        "The airplane had accumulated 15,104total flight hours before the accident flight.",
        "The pilot entered reduced visibility at tour altitude s and the accident occurred.",
        "The operator trained employees to identify signs of imp airment in passengers.",
        "The report cited StatGear 2019 and Benchmade2019 while describing the airplane.",
    ],
)
def test_builder_rejects_boilerplate_and_obvious_ocr_fragments(bad_text: str) -> None:
    from aerollm.training.sft_data import _best_sentence

    assert _best_sentence(bad_text) is None


def test_validator_quality_gate_uses_the_same_rules_as_selection() -> None:
    assert (
        _quality_rejection_reason(
            "The accident airplane landed safely after the flight crew reported an engine warning."
        )
        is None
    )
