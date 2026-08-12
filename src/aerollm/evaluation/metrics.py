"""Deterministic scoring for grounded-QA predictions."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from enum import StrEnum

from aerollm.common.schemas import GroundedAnswer
from aerollm.evaluation.schemas import GroundedQAExample


class FailureLabel(StrEnum):
    ANSWER_MISMATCH = "answer_mismatch"
    INCORRECT_CITATION = "incorrect_citation"
    INVALID_CITATION_SPAN = "invalid_citation_span"
    MISSED_ABSTENTION = "missed_abstention"
    MISSING_CITATION = "missing_citation"
    MISSING_GOLD_CITATION = "missing_gold_citation"
    UNEXPECTED_ABSTENTION = "unexpected_abstention"


@dataclass(frozen=True, slots=True)
class ExampleScore:
    example_id: str
    answer_correct: bool
    citation_precision: float
    citation_recall: float
    citation_span_validity: float
    abstention_correct: bool
    predicted_abstention: bool
    expected_abstention: bool
    failure_labels: tuple[FailureLabel, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "answer_correct": self.answer_correct,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_span_validity": self.citation_span_validity,
            "abstention_correct": self.abstention_correct,
            "predicted_abstention": self.predicted_abstention,
            "expected_abstention": self.expected_abstention,
            "failure_labels": [label.value for label in self.failure_labels],
        }


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(value: str) -> str:
    """Apply the conventional lowercase/punctuation/article normalization."""

    lowered = value.casefold()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", without_articles).strip()


def score_example(
    example: GroundedQAExample,
    prediction: GroundedAnswer,
    chunks: dict[str, str],
) -> ExampleScore:
    """Score one prediction using only exact, inspectable operations."""

    expected_abstention = not example.answerable
    abstention_correct = prediction.abstained is expected_abstention
    labels: set[FailureLabel] = set()

    if prediction.abstained:
        answer_correct = expected_abstention
        if not expected_abstention:
            labels.add(FailureLabel.UNEXPECTED_ABSTENTION)
    else:
        answer_correct = bool(
            example.reference_answer
            and prediction.answer
            and normalize_answer(prediction.answer) == normalize_answer(example.reference_answer)
        )
        if expected_abstention:
            labels.add(FailureLabel.MISSED_ABSTENTION)
        elif not answer_correct:
            labels.add(FailureLabel.ANSWER_MISMATCH)

    predicted = {(span.chunk_id, span.quote) for span in prediction.citations}
    gold = {(span.chunk_id, span.quote) for span in example.evidence}
    correct = predicted & gold
    citation_precision = len(correct) / len(predicted) if predicted else (1.0 if not gold else 0.0)
    citation_recall = len(correct) / len(gold) if gold else (1.0 if not predicted else 0.0)
    valid_count = sum(
        span.chunk_id in chunks and span.quote in chunks[span.chunk_id]
        for span in prediction.citations
    )
    span_validity = valid_count / len(prediction.citations) if prediction.citations else 1.0

    if not prediction.abstained and not prediction.citations:
        labels.add(FailureLabel.MISSING_CITATION)
    if span_validity < 1.0:
        labels.add(FailureLabel.INVALID_CITATION_SPAN)
    if citation_precision < 1.0 and prediction.citations:
        labels.add(FailureLabel.INCORRECT_CITATION)
    if citation_recall < 1.0 and gold:
        labels.add(FailureLabel.MISSING_GOLD_CITATION)

    return ExampleScore(
        example_id=example.example_id,
        answer_correct=answer_correct,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_span_validity=span_validity,
        abstention_correct=abstention_correct,
        predicted_abstention=prediction.abstained,
        expected_abstention=expected_abstention,
        failure_labels=tuple(sorted(labels, key=str)),
    )
