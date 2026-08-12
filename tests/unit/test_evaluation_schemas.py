import json

import pytest

from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample


def provenance(*, synthetic: bool = False, reviewed: bool = True) -> ExampleProvenance:
    return ExampleProvenance(
        author="analyst-a",
        reviewers=("analyst-b",) if reviewed else (),
        source_document_ids=("report-123",),
        is_synthetic=synthetic,
    )


def example(
    *,
    example_id: str = "qa-1",
    family: str = "event-family-123",
    split: Split = Split.TEST,
    example_provenance: ExampleProvenance | None = None,
) -> GroundedQAExample:
    return GroundedQAExample(
        example_id=example_id,
        question="What phase of flight was reported?",
        report_id="report-123",
        event_id="event-123",
        event_family_id=family,
        split=split,
        answerable=True,
        reference_answer="The event occurred during landing.",
        rubric=("Identifies landing as the phase of flight.",),
        evidence=(EvidenceSpan(chunk_id="report-123:p2:1", quote="during landing"),),
        provenance=example_provenance or provenance(),
    )


def test_dataset_json_round_trip_is_lossless() -> None:
    dataset = EvaluationDataset(dataset_id="grounded-qa", version="1.0.0", examples=(example(),))

    serialized = dataset.to_json()

    assert EvaluationDataset.from_json(serialized) == dataset
    assert json.loads(serialized)["examples"][0]["split"] == "test"


def test_unanswerable_example_has_rubric_but_no_answer_or_evidence() -> None:
    record = GroundedQAExample(
        example_id="qa-2",
        question="What was the pilot's motive?",
        report_id="report-123",
        event_id="event-123",
        event_family_id="event-family-123",
        split=Split.DEVELOPMENT,
        answerable=False,
        reference_answer=None,
        rubric=("Abstains because the report does not state a motive.",),
        evidence=(),
        provenance=provenance(reviewed=False),
    )

    assert record.answerable is False


@pytest.mark.parametrize(
    ("answer", "evidence", "message"),
    [
        (None, (EvidenceSpan("chunk-1", "landing"),), "reference_answer"),
        ("Landing", (), "verified evidence"),
    ],
)
def test_answerable_example_requires_answer_and_evidence(
    answer: str | None, evidence: tuple[EvidenceSpan, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GroundedQAExample(
            example_id="qa-invalid",
            question="When?",
            report_id="report-123",
            event_id="event-123",
            event_family_id="event-family-123",
            split=Split.TRAIN,
            answerable=True,
            reference_answer=answer,
            rubric=(),
            evidence=evidence,
            provenance=provenance(reviewed=False),
        )


def test_event_family_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="crosses splits"):
        EvaluationDataset(
            dataset_id="grounded-qa",
            version="1.0.0",
            examples=(
                example(example_id="qa-train", split=Split.TRAIN),
                example(example_id="qa-test", split=Split.TEST),
            ),
        )


@pytest.mark.parametrize(
    "bad_provenance",
    [provenance(synthetic=True), provenance(reviewed=False)],
)
def test_frozen_test_rejects_synthetic_or_unreviewed_examples(
    bad_provenance: ExampleProvenance,
) -> None:
    with pytest.raises(ValueError, match="frozen test"):
        EvaluationDataset(
            dataset_id="grounded-qa",
            version="1.0.0",
            examples=(example(example_provenance=bad_provenance),),
        )


def test_deserialization_rejects_unknown_fields() -> None:
    payload = EvaluationDataset(
        dataset_id="grounded-qa", version="1.0.0", examples=(example(),)
    ).to_dict()
    payload["judge_model"] = "not-supported"

    with pytest.raises(ValueError, match="unknown dataset fields: judge_model"):
        EvaluationDataset.from_dict(payload)
