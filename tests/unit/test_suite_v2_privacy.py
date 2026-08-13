import pytest

from aerollm.evaluation.suite_v2_review import (
    build_public_test_authoring_template,
    split_suite_v2_review,
    validate_public_test_manifest,
)


def _packet() -> dict[str, object]:
    common = {
        "event_family_id": "EVENT",
        "report_id": "EVENT:report",
        "task_type": "numeric",
        "scoring_strategy": "numeric_exact_or_tolerance_v1",
    }
    return {
        "schema_version": 1,
        "dataset_id": "suite-v2",
        "version": "2.0.0-draft",
        "status": "human_review_and_test_authoring_required",
        "corpus_version": "corpus-v2",
        "coverage": {"total": 2},
        "examples": [
            {
                **common,
                "example_id": "dev-1",
                "split": "development",
                "question": "How many?",
                "reference_answer": "Two.",
            },
            {
                **common,
                "example_id": "test-1",
                "split": "test",
                "question": "",
                "reference_answer": None,
                "evidence": [],
                "required_key_facts": [],
                "structured_target": None,
                "unanswerable_search_note": None,
                "author": None,
                "reviewers": [],
                "review_status": "authoring_required",
                "is_synthetic": False,
                "review_notes": "",
            },
        ],
    }


def test_split_keeps_development_labels_and_strips_test_labels() -> None:
    coverage, development, test_manifest = split_suite_v2_review(_packet())

    assert coverage["test_gold_policy"] == "private_and_ignored"
    assert development["examples"][0]["question"] == "How many?"  # type: ignore[index]
    public_test = test_manifest["examples"][0]  # type: ignore[index]
    assert set(public_test) == {
        "example_id",
        "event_family_id",
        "report_id",
        "task_type",
        "scoring_strategy",
    }
    assert "question" not in public_test
    assert "reference_answer" not in public_test
    assert "evidence" not in public_test


def test_public_test_validation_rejects_a_gold_field() -> None:
    _, _, manifest = split_suite_v2_review(_packet())
    manifest["examples"][0]["reference_answer"] = "secret"  # type: ignore[index]

    with pytest.raises(ValueError, match="non-public fields"):
        validate_public_test_manifest(manifest)


def test_public_authoring_template_exposes_blank_workflow_fields() -> None:
    template = build_public_test_authoring_template(_packet())
    example = template["examples"][0]  # type: ignore[index]

    assert template["status"] == "blank_public_authoring_template"
    assert example["question"] == ""
    assert example["review_status"] == "authoring_required"
    assert example["structured_target"] is None


def test_public_authoring_template_rejects_filled_gold() -> None:
    packet = _packet()
    packet["examples"][1]["question"] = "A leaked test question"  # type: ignore[index]

    with pytest.raises(ValueError, match="contains gold field: question"):
        build_public_test_authoring_template(packet)
