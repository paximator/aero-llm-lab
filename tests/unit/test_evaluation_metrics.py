import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.schemas import EvidenceSpan, GroundedAnswer, Split
from aerollm.evaluation.cli import main
from aerollm.evaluation.metrics import FailureLabel, normalize_answer, score_example
from aerollm.evaluation.runner import PredictionRecord, evaluate, predictions_from_json
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample

FIXTURES = Path(__file__).parents[1] / "fixtures" / "evaluation"


def answerable_example() -> GroundedQAExample:
    return GroundedQAExample(
        example_id="qa-1",
        question="What happened?",
        report_id="report-1",
        event_id="event-1",
        event_family_id="family-1",
        split=Split.DEVELOPMENT,
        answerable=True,
        reference_answer="The aircraft landed.",
        rubric=("Identifies landing.",),
        evidence=(EvidenceSpan("chunk-1", "aircraft landed"),),
        provenance=ExampleProvenance("author", (), ("report-1",)),
    )


def test_answer_normalization_is_deterministic() -> None:
    assert normalize_answer("  The aircraft, LANDED! ") == "aircraft landed"


def test_scoring_detects_invalid_and_incorrect_citation() -> None:
    prediction = GroundedAnswer(
        answer="The aircraft landed.",
        citations=(EvidenceSpan("chunk-1", "invented quote"),),
        abstained=False,
    )

    score = score_example(answerable_example(), prediction, {"chunk-1": "The aircraft landed."})

    assert score.answer_correct
    assert score.citation_precision == 0.0
    assert score.citation_recall == 0.0
    assert score.citation_span_validity == 0.0
    assert set(score.failure_labels) == {
        FailureLabel.INCORRECT_CITATION,
        FailureLabel.INVALID_CITATION_SPAN,
        FailureLabel.MISSING_GOLD_CITATION,
    }


def test_prediction_parser_rejects_non_boolean_abstained() -> None:
    payload = json.loads((FIXTURES / "predictions_v1.json").read_text(encoding="utf-8"))
    payload[0]["output"]["abstained"] = 0

    with pytest.raises(ValueError, match="abstained must be a boolean"):
        predictions_from_json(json.dumps(payload))


def test_evaluate_requires_exact_prediction_coverage() -> None:
    dataset = EvaluationDataset(dataset_id="test", version="1", examples=(answerable_example(),))

    with pytest.raises(ValueError, match=r"missing=\['qa-1'\]"):
        evaluate(dataset, ())


def test_fixture_evaluation_has_perfect_scores_and_slices() -> None:
    dataset_text = (FIXTURES / "grounded_qa_v1.json").read_text(encoding="utf-8")
    predictions_text = (FIXTURES / "predictions_v1.json").read_text(encoding="utf-8")

    report = evaluate(
        EvaluationDataset.from_json(dataset_text), predictions_from_json(predictions_text)
    )

    assert report.aggregate.answer_accuracy == 1.0
    assert report.aggregate.abstention_precision == 1.0
    assert report.aggregate.abstention_recall == 1.0
    assert set(report.slices) == {"answerable:true", "answerable:false", "split:test"}


def test_frozen_fixture_matches_manifest_digest() -> None:
    manifest = json.loads((FIXTURES / "manifest_v1.json").read_text(encoding="utf-8"))
    dataset_bytes = (FIXTURES / manifest["path"]).read_bytes()

    assert manifest["frozen"] is True
    assert hashlib.sha256(dataset_bytes).hexdigest() == manifest["sha256"]


def test_cli_writes_versioned_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    result = main(
        [
            str(FIXTURES / "grounded_qa_v1.json"),
            str(FIXTURES / "predictions_v1.json"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["dataset_version"] == "1.0.0"
    assert len(report["dataset_sha256"]) == 64
    assert report["evaluator"] == "deterministic-v1"


def test_prediction_record_rejects_unknown_model_judge_field() -> None:
    payload = json.loads((FIXTURES / "predictions_v1.json").read_text(encoding="utf-8"))[0]
    payload["judge_score"] = 0.9

    with pytest.raises(ValueError, match="judge_score"):
        PredictionRecord.from_dict(payload)
