import json
from pathlib import Path

import pytest

from aerollm.evaluation.task_scoring import (
    apply_target_config,
    evaluate_task_answers,
    score_task_answer,
    validate_task_annotation,
)
from aerollm.evaluation.task_scoring_cli import main


def _example(strategy: str, target: object, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "example_id": "example-1",
        "task_type": "numeric",
        "scoring_strategy": strategy,
        "structured_target": target,
        "required_key_facts": [],
        "answerable": True,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("example", "answer"),
    [
        (
            _example(
                "numeric_exact_or_tolerance_v1",
                {"value": 2700, "unit": "ft_msl", "tolerance": 50},
            ),
            "The altitude was about 2,675 ft msl.",
        ),
        (
            _example(
                "normalized_datetime_v1",
                {
                    "date": "2015-11-10",
                    "local_time": "14:53",
                    "timezone_label": "EST",
                    "approximate": True,
                },
                task_type="date_time",
            ),
            "November 10, 2015 at 1453 EST.",
        ),
        (
            _example(
                "normalized_entity_set_v1",
                {"required": ["Boeing 747-400 BCF", "N949CA"]},
                task_type="entity_identifier",
            ),
            "It was a Boeing 747-400 BCF registered N949CA.",
        ),
        (
            _example("categorical_exact_v1", {"label": "yes"}, task_type="categorical_yes_no"),
            "Yes.",
        ),
        (
            _example(
                "required_key_facts_v1",
                None,
                task_type="causal_finding",
                required_key_facts=["cargo was not properly restrained"],
            ),
            "The cargo was not properly restrained.",
        ),
        (
            _example("abstention_v1", None, task_type="unanswerable", answerable=False),
            None,
        ),
    ],
)
def test_each_v2_strategy_scores_a_correct_answer(
    example: dict[str, object], answer: str | None
) -> None:
    assert score_task_answer(example, answer)


def test_validation_rejects_declared_scorer_without_target() -> None:
    with pytest.raises(ValueError, match="requires structured_target"):
        validate_task_annotation(_example("normalized_entity_set_v1", None))


def test_validation_rejects_empty_required_key_facts() -> None:
    with pytest.raises(ValueError, match="non-empty required_key_facts"):
        validate_task_annotation(_example("required_key_facts_v1", None))


def test_report_contains_per_task_metrics() -> None:
    examples = [
        _example(
            "numeric_exact_or_tolerance_v1",
            {"value": 9, "unit": "people", "tolerance": 0},
        ),
        _example(
            "abstention_v1",
            None,
            example_id="example-2",
            task_type="unanswerable",
            answerable=False,
        ),
    ]

    report = evaluate_task_answers(examples, {"example-1": "Nine people.", "example-2": None})

    assert report["accuracy"] == 1.0
    assert report["by_task_type"] == {
        "numeric": {"count": 1, "accuracy": 1.0},
        "unanswerable": {"count": 1, "accuracy": 1.0},
    }


def test_target_config_is_complete_for_public_development_data() -> None:
    root = Path(__file__).parents[2]
    dataset = json.loads((root / "data/evaluation/v2/development.json").read_text(encoding="utf-8"))

    apply_target_config(dataset, root / "configs/evaluation/evaluation_suite_v2.targets.toml")

    assert len(dataset["examples"]) == 27
    assert all(
        score_task_answer(example, example.get("reference_answer"))
        for example in dataset["examples"]
    )


def test_cli_validates_public_development_data() -> None:
    root = Path(__file__).parents[2]

    assert main([str(root / "data/evaluation/v2/development.json")]) == 0
