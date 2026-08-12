"""Dataset-level deterministic evaluation and JSON input records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from aerollm.common.schemas import EvidenceSpan, GroundedAnswer
from aerollm.evaluation.metrics import ExampleScore, score_example
from aerollm.evaluation.schemas import EvaluationDataset


def _strict_keys(value: Mapping[str, object], expected: set[str], kind: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        raise ValueError(f"invalid {kind} fields ({'; '.join(details)})")


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    example_id: str
    output: GroundedAnswer
    chunks: dict[str, str]

    _FIELDS: ClassVar[set[str]] = {"example_id", "output", "chunks"}
    _OUTPUT_FIELDS: ClassVar[set[str]] = {
        "answer",
        "citations",
        "abstained",
        "abstention_reason",
    }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PredictionRecord:
        if not isinstance(value, Mapping):
            raise ValueError("prediction must be an object")
        _strict_keys(value, cls._FIELDS, "prediction")
        example_id = value["example_id"]
        output = value["output"]
        chunks = value["chunks"]
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(output, Mapping):
            raise ValueError("output must be an object")
        _strict_keys(output, cls._OUTPUT_FIELDS, "output")
        citations = output["citations"]
        if not isinstance(citations, list):
            raise ValueError("citations must be a list")
        spans = []
        for citation in citations:
            if not isinstance(citation, Mapping):
                raise ValueError("citation must be an object")
            _strict_keys(citation, {"chunk_id", "quote"}, "citation")
            chunk_id, quote = citation["chunk_id"], citation["quote"]
            if not isinstance(chunk_id, str) or not isinstance(quote, str):
                raise ValueError("citation chunk_id and quote must be strings")
            spans.append(EvidenceSpan(chunk_id=chunk_id, quote=quote))
        if not isinstance(chunks, Mapping) or not all(
            isinstance(key, str) and isinstance(text, str) for key, text in chunks.items()
        ):
            raise ValueError("chunks must be an object mapping IDs to text")
        answer = output["answer"]
        reason = output["abstention_reason"]
        abstained = output["abstained"]
        if answer is not None and not isinstance(answer, str):
            raise ValueError("answer must be a string or null")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("abstention_reason must be a string or null")
        if type(abstained) is not bool:
            raise ValueError("abstained must be a boolean")
        return cls(
            example_id=example_id,
            output=GroundedAnswer(answer, tuple(spans), abstained, reason),
            chunks=dict(chunks),
        )


@dataclass(frozen=True, slots=True)
class AggregateScore:
    count: int
    answer_accuracy: float
    citation_precision: float
    citation_recall: float
    citation_span_validity: float
    abstention_accuracy: float
    abstention_precision: float
    abstention_recall: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "answer_accuracy": self.answer_accuracy,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_span_validity": self.citation_span_validity,
            "abstention_accuracy": self.abstention_accuracy,
            "abstention_precision": self.abstention_precision,
            "abstention_recall": self.abstention_recall,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_id: str
    dataset_version: str
    dataset_sha256: str
    scores: tuple[ExampleScore, ...]
    aggregate: AggregateScore
    slices: dict[str, AggregateScore]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "scores": [score.to_dict() for score in self.scores],
            "aggregate": self.aggregate.to_dict(),
            "slices": {name: score.to_dict() for name, score in sorted(self.slices.items())},
            "evaluator": "deterministic-v1",
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _aggregate(scores: Sequence[ExampleScore]) -> AggregateScore:
    count = len(scores)
    if not count:
        raise ValueError("cannot aggregate an empty score collection")
    predicted_abstentions = sum(score.predicted_abstention for score in scores)
    expected_abstentions = sum(score.expected_abstention for score in scores)
    correct_abstentions = sum(
        score.predicted_abstention and score.expected_abstention for score in scores
    )
    return AggregateScore(
        count=count,
        answer_accuracy=sum(score.answer_correct for score in scores) / count,
        citation_precision=sum(score.citation_precision for score in scores) / count,
        citation_recall=sum(score.citation_recall for score in scores) / count,
        citation_span_validity=sum(score.citation_span_validity for score in scores) / count,
        abstention_accuracy=sum(score.abstention_correct for score in scores) / count,
        abstention_precision=(
            correct_abstentions / predicted_abstentions if predicted_abstentions else 1.0
        ),
        abstention_recall=(
            correct_abstentions / expected_abstentions if expected_abstentions else 1.0
        ),
    )


def evaluate(
    dataset: EvaluationDataset,
    predictions: Sequence[PredictionRecord],
    *,
    dataset_bytes: bytes | None = None,
) -> EvaluationReport:
    """Evaluate exactly one prediction for every dataset example."""

    by_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.example_id in by_id:
            raise ValueError(f"duplicate prediction: {prediction.example_id}")
        by_id[prediction.example_id] = prediction
    expected = {example.example_id for example in dataset.examples}
    missing, extra = expected - set(by_id), set(by_id) - expected
    if missing or extra:
        detail = f"missing={sorted(missing)}, extra={sorted(extra)}"
        raise ValueError(f"prediction IDs do not match dataset ({detail})")

    scores = tuple(
        score_example(example, by_id[example.example_id].output, by_id[example.example_id].chunks)
        for example in dataset.examples
    )
    groups: defaultdict[str, list[ExampleScore]] = defaultdict(list)
    for example, score in zip(dataset.examples, scores, strict=True):
        groups[f"answerable:{str(example.answerable).lower()}"].append(score)
        groups[f"split:{example.split.value}"].append(score)
    canonical = dataset.to_json().encode() if dataset_bytes is None else dataset_bytes
    return EvaluationReport(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_sha256=hashlib.sha256(canonical).hexdigest(),
        scores=scores,
        aggregate=_aggregate(scores),
        slices={name: _aggregate(group) for name, group in groups.items()},
    )


def predictions_from_json(value: str) -> tuple[PredictionRecord, ...]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("predictions are not valid JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError("predictions must be a JSON list")
    return tuple(PredictionRecord.from_dict(item) for item in decoded)
