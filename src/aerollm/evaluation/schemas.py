"""Strict, serializable records for the initial grounded-QA evaluation suite."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from aerollm.common.schemas import EvidenceSpan, Split


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_keys(value: Mapping[str, object], expected: set[str], kind: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"unknown {kind} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {kind} fields: {', '.join(sorted(missing))}")


@dataclass(frozen=True, slots=True)
class ExampleProvenance:
    """Human authorship and review trail for a gold evaluation example."""

    author: str
    reviewers: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    is_synthetic: bool = False

    _FIELDS: ClassVar[set[str]] = {
        "author",
        "reviewers",
        "source_document_ids",
        "is_synthetic",
    }

    def __post_init__(self) -> None:
        _require_string(self.author, "author")
        if type(self.is_synthetic) is not bool:
            raise ValueError("is_synthetic must be a boolean")
        if not isinstance(self.reviewers, tuple):
            raise ValueError("reviewers must be a tuple")
        for reviewer in self.reviewers:
            _require_string(reviewer, "reviewer")
        if self.author in self.reviewers:
            raise ValueError("an example must be independently reviewed")
        if len(set(self.reviewers)) != len(self.reviewers):
            raise ValueError("reviewers must be unique")
        if not isinstance(self.source_document_ids, tuple) or not self.source_document_ids:
            raise ValueError("at least one source_document_id is required")
        for source_id in self.source_document_ids:
            _require_string(source_id, "source_document_id")
        if len(set(self.source_document_ids)) != len(self.source_document_ids):
            raise ValueError("source_document_ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "author": self.author,
            "reviewers": list(self.reviewers),
            "source_document_ids": list(self.source_document_ids),
            "is_synthetic": self.is_synthetic,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExampleProvenance:
        if not isinstance(value, Mapping):
            raise ValueError("provenance must be an object")
        _require_keys(value, cls._FIELDS, "provenance")
        reviewers = value["reviewers"]
        source_ids = value["source_document_ids"]
        if not isinstance(reviewers, list) or not all(isinstance(item, str) for item in reviewers):
            raise ValueError("reviewers must be a list of strings")
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) for item in source_ids
        ):
            raise ValueError("source_document_ids must be a list of strings")
        return cls(
            author=value["author"],  # type: ignore[arg-type]
            reviewers=tuple(reviewers),
            source_document_ids=tuple(source_ids),
            is_synthetic=value["is_synthetic"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GroundedQAExample:
    """One human-verifiable question over a single aviation report event."""

    example_id: str
    question: str
    report_id: str
    event_id: str
    event_family_id: str
    split: Split
    answerable: bool
    reference_answer: str | None
    rubric: tuple[str, ...]
    evidence: tuple[EvidenceSpan, ...]
    provenance: ExampleProvenance

    _FIELDS: ClassVar[set[str]] = {
        "example_id",
        "question",
        "report_id",
        "event_id",
        "event_family_id",
        "split",
        "answerable",
        "reference_answer",
        "rubric",
        "evidence",
        "provenance",
    }

    def __post_init__(self) -> None:
        for name in ("example_id", "question", "report_id", "event_id", "event_family_id"):
            _require_string(getattr(self, name), name)
        if not isinstance(self.split, Split):
            raise ValueError("split must be a Split")
        if type(self.answerable) is not bool:
            raise ValueError("answerable must be a boolean")
        if self.reference_answer is not None and not isinstance(self.reference_answer, str):
            raise ValueError("reference_answer must be a string or null")
        if not isinstance(self.rubric, tuple):
            raise ValueError("rubric must be a tuple")
        for item in self.rubric:
            _require_string(item, "rubric item")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(span, EvidenceSpan) for span in self.evidence
        ):
            raise ValueError("evidence must be a tuple of EvidenceSpan records")
        if not isinstance(self.provenance, ExampleProvenance):
            raise ValueError("provenance must be an ExampleProvenance")
        if self.answerable:
            _require_string(self.reference_answer, "reference_answer")
            if not self.evidence:
                raise ValueError("an answerable example requires verified evidence")
        elif self.reference_answer is not None or self.evidence:
            raise ValueError("an unanswerable example cannot have an answer or evidence")
        if not self.reference_answer and not self.rubric:
            raise ValueError("an example requires a reference answer or rubric")

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "report_id": self.report_id,
            "event_id": self.event_id,
            "event_family_id": self.event_family_id,
            "split": self.split.value,
            "answerable": self.answerable,
            "reference_answer": self.reference_answer,
            "rubric": list(self.rubric),
            "evidence": [
                {"chunk_id": span.chunk_id, "quote": span.quote} for span in self.evidence
            ],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GroundedQAExample:
        if not isinstance(value, Mapping):
            raise ValueError("example must be an object")
        _require_keys(value, cls._FIELDS, "example")
        rubric = value["rubric"]
        evidence = value["evidence"]
        if not isinstance(rubric, list) or not all(isinstance(item, str) for item in rubric):
            raise ValueError("rubric must be a list of strings")
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        spans: list[EvidenceSpan] = []
        for item in evidence:
            if not isinstance(item, Mapping):
                raise ValueError("each evidence item must be an object")
            _require_keys(item, {"chunk_id", "quote"}, "evidence")
            spans.append(
                EvidenceSpan(
                    chunk_id=_require_string(item["chunk_id"], "chunk_id"),
                    quote=_require_string(item["quote"], "quote"),
                )
            )
        try:
            split = Split(value["split"])
        except (TypeError, ValueError) as exc:
            raise ValueError("split must be train, development, or test") from exc
        provenance = value["provenance"]
        return cls(
            example_id=value["example_id"],  # type: ignore[arg-type]
            question=value["question"],  # type: ignore[arg-type]
            report_id=value["report_id"],  # type: ignore[arg-type]
            event_id=value["event_id"],  # type: ignore[arg-type]
            event_family_id=value["event_family_id"],  # type: ignore[arg-type]
            split=split,
            answerable=value["answerable"],  # type: ignore[arg-type]
            reference_answer=value["reference_answer"],  # type: ignore[arg-type]
            rubric=tuple(rubric),
            evidence=tuple(spans),
            provenance=ExampleProvenance.from_dict(provenance),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned collection whose splits are assigned at event-family level."""

    dataset_id: str
    version: str
    examples: tuple[GroundedQAExample, ...]

    _FIELDS: ClassVar[set[str]] = {"dataset_id", "version", "examples"}

    def __post_init__(self) -> None:
        _require_string(self.dataset_id, "dataset_id")
        _require_string(self.version, "version")
        if not isinstance(self.examples, tuple) or not self.examples:
            raise ValueError("an evaluation dataset requires examples")
        ids: set[str] = set()
        family_splits: dict[str, Split] = {}
        for example in self.examples:
            if not isinstance(example, GroundedQAExample):
                raise ValueError("examples must contain GroundedQAExample records")
            if example.example_id in ids:
                raise ValueError(f"duplicate example_id: {example.example_id}")
            ids.add(example.example_id)
            previous = family_splits.setdefault(example.event_family_id, example.split)
            if previous is not example.split:
                raise ValueError(
                    f"event family {example.event_family_id!r} crosses splits: "
                    f"{previous.value} and {example.split.value}"
                )
            if example.split is Split.TEST:
                if example.provenance.is_synthetic:
                    raise ValueError("frozen test examples cannot be synthetic")
                if not example.provenance.reviewers:
                    raise ValueError("frozen test examples require independent human review")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "examples": [example.to_dict() for example in self.examples],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EvaluationDataset:
        if not isinstance(value, Mapping):
            raise ValueError("dataset must be an object")
        _require_keys(value, cls._FIELDS, "dataset")
        examples = value["examples"]
        if not isinstance(examples, list):
            raise ValueError("examples must be a list")
        return cls(
            dataset_id=value["dataset_id"],  # type: ignore[arg-type]
            version=value["version"],  # type: ignore[arg-type]
            examples=tuple(GroundedQAExample.from_dict(item) for item in examples),  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, value: str) -> EvaluationDataset:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("dataset is not valid JSON") from exc
        return cls.from_dict(decoded)
