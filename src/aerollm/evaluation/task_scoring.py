"""Small, deterministic scorer set for evaluation-suite-v2 development data."""

from __future__ import annotations

import json
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from aerollm.evaluation.metrics import normalize_answer

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}
_STRATEGIES = {
    "numeric_exact_or_tolerance_v1",
    "normalized_datetime_v1",
    "normalized_entity_set_v1",
    "categorical_exact_v1",
    "required_key_facts_v1",
    "abstention_v1",
}


def apply_target_config(dataset: dict[str, object], config_path: Path) -> None:
    """Apply separately reviewed targets to a mutable public development dataset."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    targets = config.get("targets", {})
    facts = config.get("required_key_facts", {})
    if not isinstance(targets, Mapping) or not isinstance(facts, Mapping):
        raise ValueError("target configuration sections must be tables")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise ValueError("development examples must be a list")
    known_ids = {item.get("example_id") for item in examples if isinstance(item, Mapping)}
    unknown = (set(targets) | set(facts)) - known_ids
    if unknown:
        raise ValueError(f"target configuration references unknown examples: {sorted(unknown)}")
    for item in examples:
        if not isinstance(item, dict):
            raise ValueError("development examples must be objects")
        example_id = item.get("example_id")
        if example_id in targets:
            item["structured_target"] = dict(targets[example_id])
        if example_id in facts:
            item["required_key_facts"] = list(facts[example_id])
        validate_task_annotation(item)


def validate_task_annotation(example: Mapping[str, object]) -> None:
    """Require the target shape declared by an example's scoring strategy."""
    strategy = example.get("scoring_strategy")
    if strategy not in _STRATEGIES:
        raise ValueError(f"unsupported scoring strategy: {strategy}")
    target = example.get("structured_target")
    facts = example.get("required_key_facts")
    if strategy == "numeric_exact_or_tolerance_v1":
        _require_target(target, {"value", "unit", "tolerance"}, strategy)
        assert isinstance(target, Mapping)
        if isinstance(target["value"], bool) or not isinstance(target["value"], int | float):
            raise ValueError("numeric target value must be a number")
        if not isinstance(target["tolerance"], int | float) or target["tolerance"] < 0:
            raise ValueError("numeric target tolerance must be non-negative")
        _nonempty_string(target["unit"], "numeric target unit")
    elif strategy == "normalized_datetime_v1":
        _require_target(target, {"date", "local_time", "timezone_label", "approximate"}, strategy)
        assert isinstance(target, Mapping)
        for field in ("date", "local_time", "timezone_label"):
            _nonempty_string(target[field], f"datetime target {field}")
        if type(target["approximate"]) is not bool:
            raise ValueError("datetime target approximate must be a boolean")
    elif strategy == "normalized_entity_set_v1":
        _require_target(target, {"required"}, strategy)
        assert isinstance(target, Mapping)
        required = target["required"]
        if not isinstance(required, list) or not required:
            raise ValueError("entity target required must be a non-empty list")
        for entity in required:
            _nonempty_string(entity, "required entity")
    elif strategy == "categorical_exact_v1":
        _require_target(target, {"label"}, strategy)
        assert isinstance(target, Mapping)
        _nonempty_string(target["label"], "categorical target label")
    elif strategy == "required_key_facts_v1":
        if not isinstance(facts, list) or not facts:
            raise ValueError("required_key_facts_v1 requires non-empty required_key_facts")
        for fact in facts:
            _nonempty_string(fact, "required key fact")
    elif strategy == "abstention_v1" and example.get("answerable") is not False:
        raise ValueError("abstention_v1 requires answerable=false")


def score_task_answer(example: Mapping[str, object], answer: str | None) -> bool:
    """Score one answer with the example's validated deterministic strategy."""
    validate_task_annotation(example)
    strategy = str(example["scoring_strategy"])
    if strategy == "abstention_v1":
        return answer is None or not answer.strip()
    if answer is None or not answer.strip():
        return False
    target = example.get("structured_target")
    if strategy == "numeric_exact_or_tolerance_v1":
        assert isinstance(target, Mapping)
        numbers = [float(value.replace(",", "")) for value in _NUMBER.findall(answer)]
        normalized_words = set(normalize_answer(answer).split())
        numbers.extend(value for word, value in _NUMBER_WORDS.items() if word in normalized_words)
        expected = float(target["value"])  # type: ignore[arg-type]
        tolerance = float(target["tolerance"])  # type: ignore[arg-type]
        return any(abs(value - expected) <= tolerance for value in numbers)
    normalized = normalize_answer(answer)
    if strategy == "normalized_datetime_v1":
        assert isinstance(target, Mapping)
        return _date_matches(normalized, target)
    if strategy == "normalized_entity_set_v1":
        assert isinstance(target, Mapping)
        return all(normalize_answer(entity) in normalized for entity in target["required"])  # type: ignore[union-attr]
    if strategy == "categorical_exact_v1":
        assert isinstance(target, Mapping)
        return normalized == normalize_answer(str(target["label"]))
    facts = example["required_key_facts"]
    assert isinstance(facts, list)
    return all(normalize_answer(fact) in normalized for fact in facts)


def evaluate_task_answers(
    examples: Sequence[Mapping[str, object]], predictions: Mapping[str, str | None]
) -> dict[str, object]:
    """Return aggregate and per-task deterministic accuracy."""
    expected = {str(example["example_id"]) for example in examples}
    if set(predictions) != expected:
        raise ValueError("prediction IDs must exactly match development example IDs")
    by_task: defaultdict[str, list[bool]] = defaultdict(list)
    scores: list[dict[str, object]] = []
    for example in examples:
        example_id = str(example["example_id"])
        correct = score_task_answer(example, predictions[example_id])
        task = str(example["task_type"])
        by_task[task].append(correct)
        scores.append({"example_id": example_id, "task_type": task, "correct": correct})
    return {
        "evaluator": "task-specific-v2",
        "count": len(scores),
        "accuracy": sum(item["correct"] for item in scores) / len(scores),
        "by_task_type": {
            task: {"count": len(values), "accuracy": sum(values) / len(values)}
            for task, values in sorted(by_task.items())
        },
        "scores": scores,
    }


def write_enriched_development(dataset_path: Path, config_path: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("development dataset must be a JSON object")
    apply_target_config(dataset, config_path)
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_target(target: object, fields: set[str], strategy: str) -> None:
    if not isinstance(target, Mapping) or set(target) != fields:
        raise ValueError(f"{strategy} requires structured_target fields {sorted(fields)}")


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _date_matches(answer: str, target: Mapping[str, object]) -> bool:
    date = str(target["date"])
    year, month, day = date.split("-")
    time = str(target["local_time"])
    hour, minute = time.split(":")
    month_names = (
        "january february march april may june july august september october november december"
    ).split()
    date_forms = {normalize_answer(date), f"{month_names[int(month) - 1]} {int(day)} {year}"}
    time_forms = {time, f"{hour}{minute}", f"{int(hour)}{minute}"}
    return any(form in answer for form in date_forms) and any(form in answer for form in time_forms)
