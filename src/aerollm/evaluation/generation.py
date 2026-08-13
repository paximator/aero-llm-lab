"""Transparent metrics and records for comparable generation experiments."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

from aerollm.evaluation.schemas import GroundedQAExample
from aerollm.generation.backends import GenerationResult

_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_CITATION = re.compile(r"\[(\d+)]")


@dataclass(frozen=True, slots=True)
class GenerationExampleResult:
    example_id: str
    variant: str
    question: str
    reference_answer: str | None
    answerable: bool
    answer: str
    retrieved_chunk_ids: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]
    exact_match: float | None
    token_f1: float | None
    gold_evidence_coverage: float | None
    citation_precision: float | None
    citation_recall: float | None
    refusal_correct: float | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["retrieved_chunk_ids"] = list(self.retrieved_chunk_ids)
        value["context_chunk_ids"] = list(self.context_chunk_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> GenerationExampleResult:
        payload = dict(value)
        payload["retrieved_chunk_ids"] = tuple(payload["retrieved_chunk_ids"])
        payload["context_chunk_ids"] = tuple(payload["context_chunk_ids"])
        payload.setdefault("answerable", payload["reference_answer"] is not None)
        answerable = bool(payload["answerable"])
        if payload["variant"] != "rag":
            payload["gold_evidence_coverage"] = None
        if not answerable:
            payload["citation_precision"] = None
            payload["citation_recall"] = None
        payload.setdefault(
            "refusal_correct",
            float(_is_refusal(str(payload["answer"]))) if not answerable else None,
        )
        return cls(**payload)  # type: ignore[arg-type]


def score_generation(
    example: GroundedQAExample,
    variant: str,
    result: GenerationResult,
    *,
    retrieved_chunk_ids: tuple[str, ...] = (),
    context_chunk_ids: tuple[str, ...] = (),
) -> GenerationExampleResult:
    reference = example.reference_answer
    exact = token_f1 = None
    if reference is not None:
        exact = float(_normalize(result.text) == _normalize(reference))
        token_f1 = _token_f1(result.text, reference)
    gold = {span.chunk_id for span in example.evidence}
    coverage = (
        len(gold.intersection(retrieved_chunk_ids)) / len(gold)
        if gold and variant == "rag" else None
    )
    cited = {int(value) for value in _CITATION.findall(result.text)}
    valid = {value for value in cited if 1 <= value <= len(context_chunk_ids)}
    citation_precision = (
        len(valid) / len(cited) if cited else (0.0 if context_chunk_ids else None)
    ) if example.answerable else None
    citation_recall = (
        float(bool(valid)) if context_chunk_ids else None
    ) if example.answerable else None
    refusal_correct = float(_is_refusal(result.text)) if not example.answerable else None
    return GenerationExampleResult(
        example.example_id, variant, example.question, reference, example.answerable, result.text,
        retrieved_chunk_ids, context_chunk_ids, exact, token_f1, coverage,
        citation_precision, citation_recall, refusal_correct, result.prompt_tokens,
        result.completion_tokens, result.latency_ms,
    )


def aggregate_generation(scores: list[GenerationExampleResult]) -> dict[str, object]:
    if not scores:
        raise ValueError("cannot aggregate empty generation scores")
    variants: dict[str, dict[str, object]] = {}
    for variant in sorted({score.variant for score in scores}):
        group = [score for score in scores if score.variant == variant]
        variants[variant] = {
            "examples": len(group),
            "exact_match": _mean(score.exact_match for score in group),
            "token_f1": _mean(score.token_f1 for score in group),
            "gold_evidence_coverage": _mean(score.gold_evidence_coverage for score in group),
            "citation_precision": _mean(score.citation_precision for score in group),
            "citation_recall": _mean(score.citation_recall for score in group),
            "refusal_accuracy": _mean(score.refusal_correct for score in group),
            "mean_latency_ms": _mean(score.latency_ms for score in group),
            "completion_tokens": sum(score.completion_tokens for score in group),
        }
    return {"variants": variants}


def review_packet(scores: list[GenerationExampleResult]) -> dict[str, object]:
    return {
        "instructions": (
            "Review correctness, evidence support, completeness, and harmful hallucination. "
            "Do not tune using frozen test examples."
        ),
        "examples": [
            {
                **score.to_dict(),
                "human_review": {
                    "correct": None,
                    "evidence_supported": None,
                    "complete": None,
                    "notes": "",
                },
            }
            for score in scores
        ],
    }


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOKEN.findall(normalized))


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalize(prediction).split()
    expected = _normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _is_refusal(value: str) -> bool:
    normalized = _normalize(value)
    markers = ("do not know", "does not specify", "not specified", "insufficient")
    return any(marker in normalized for marker in markers)


def _mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
