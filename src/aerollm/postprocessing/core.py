"""Deterministic, fail-closed postprocessing for grounded answers."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence

from aerollm.postprocessing.schemas import (
    Citation,
    ProcessedAnswer,
    RawGeneration,
    RetrievedEvidence,
)

DEFAULT_MAX_OUTPUT_CHARS = 16_384
INSUFFICIENT_EVIDENCE = "insufficient evidence"

_JSON_FIELDS = {"answer", "citations", "abstained", "abstention_reason"}
_CITATION_FIELDS = {"chunk_id", "quote"}
_FENCE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL | re.IGNORECASE)


def normalize_whitespace(value: str) -> str:
    """Normalize Unicode whitespace to stable ASCII spaces and newlines."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in value.split("\n"):
        line = "".join(" " if character.isspace() else character for character in line)
        lines.append(re.sub(r" +", " ", line).strip())
    return "\n".join(lines).strip()


def postprocess(
    raw: RawGeneration | str,
    evidence: Sequence[RetrievedEvidence],
    *,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ProcessedAnswer:
    """Turn model output into a validated answer, or a safe abstention.

    JSON is the grounded format. Plain text is accepted safely but cannot establish
    grounding, so it deterministically abstains while preserving the normalized text
    in a warning-free audit record.
    """

    if type(max_output_chars) is not int or max_output_chars < 1:
        raise ValueError("max_output_chars must be a positive integer")
    raw_output = raw.text if isinstance(raw, RawGeneration) else raw
    if not isinstance(raw_output, str):
        raise TypeError("raw must be RawGeneration or str")
    chunks = _evidence_map(evidence)
    if not raw_output.strip():
        return _abstain(raw_output, "empty_output")
    if len(raw_output) > max_output_chars:
        return _abstain(raw_output, "output_too_long")

    normalized = normalize_whitespace(raw_output)
    fenced = _FENCE.fullmatch(normalized)
    candidate = fenced.group(1) if fenced else normalized
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, RecursionError):
        return _abstain(raw_output, "invalid_json")
    try:
        return _process_payload(payload, chunks, raw_output)
    except (TypeError, ValueError, KeyError):
        return _abstain(raw_output, "malformed_output")


def _evidence_map(evidence: Sequence[RetrievedEvidence]) -> dict[str, str]:
    chunks: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, RetrievedEvidence):
            raise TypeError("evidence must contain RetrievedEvidence records")
        if item.chunk_id in chunks:
            raise ValueError(f"duplicate evidence chunk_id: {item.chunk_id}")
        chunks[item.chunk_id] = normalize_whitespace(item.text)
    return chunks


def _process_payload(
    payload: object, chunks: Mapping[str, str], raw_output: str
) -> ProcessedAnswer:
    if not isinstance(payload, dict) or set(payload) != _JSON_FIELDS:
        raise ValueError("invalid output fields")
    if type(payload["abstained"]) is not bool:
        raise TypeError("abstained must be boolean")
    answer, reason, raw_citations = (
        payload["answer"],
        payload["abstention_reason"],
        payload["citations"],
    )
    if answer is not None and not isinstance(answer, str):
        raise TypeError("answer must be a string or null")
    if reason is not None and not isinstance(reason, str):
        raise TypeError("abstention_reason must be a string or null")
    if not isinstance(raw_citations, list):
        raise TypeError("citations must be a list")

    if payload["abstained"]:
        if answer is not None or raw_citations:
            raise ValueError("invalid abstention")
        normalized_reason = normalize_whitespace(reason or "")
        return ProcessedAnswer(
            None, (), True, normalized_reason or INSUFFICIENT_EVIDENCE, raw_output
        )

    normalized_answer = normalize_whitespace(answer or "")
    citations: list[Citation] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_citations:
        if not isinstance(item, dict) or set(item) != _CITATION_FIELDS:
            raise ValueError("invalid citation")
        chunk_id, quote = item["chunk_id"], item["quote"]
        if not isinstance(chunk_id, str) or not isinstance(quote, str):
            raise TypeError("citation fields must be strings")
        normalized_id = normalize_whitespace(chunk_id)
        normalized_quote = normalize_whitespace(quote)
        if not normalized_id or not normalized_quote or normalized_quote not in chunks.get(
            normalized_id, ""
        ):
            warnings.append("invalid_citation")
            continue
        key = (normalized_id, normalized_quote)
        if key not in seen:
            citations.append(Citation(*key))
            seen.add(key)

    if not normalized_answer or not citations:
        warning = "insufficient_evidence"
        return _abstain(raw_output, warning, tuple(dict.fromkeys((*warnings, warning))))
    return ProcessedAnswer(
        normalized_answer,
        tuple(citations),
        False,
        None,
        raw_output,
        tuple(dict.fromkeys(warnings)),
    )


def _abstain(
    raw_output: str, warning: str, warnings: tuple[str, ...] | None = None
) -> ProcessedAnswer:
    return ProcessedAnswer(
        answer=None,
        citations=(),
        abstained=True,
        abstention_reason=INSUFFICIENT_EVIDENCE,
        raw_output=raw_output,
        warnings=warnings or (warning,),
    )


class PostprocessorV1:
    """Configurable callable wrapper around :func:`postprocess`."""

    def __init__(self, *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> None:
        if type(max_output_chars) is not int or max_output_chars < 1:
            raise ValueError("max_output_chars must be a positive integer")
        self.max_output_chars = max_output_chars

    def process(
        self, raw: RawGeneration | str, evidence: Sequence[RetrievedEvidence]
    ) -> ProcessedAnswer:
        return postprocess(raw, evidence, max_output_chars=self.max_output_chars)

    __call__ = process
