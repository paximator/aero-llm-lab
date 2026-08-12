"""Validate a human-reviewed packet and freeze it as a gold test dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.data.snapshots import atomic_write
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample

_PACKET_FIELDS = {
    "schema_version", "dataset_id", "version", "author", "report_id", "split",
    "reviewer", "examples",
}
_EXAMPLE_FIELDS = {
    "example_id", "question", "reference_answer", "chunk_id", "page",
    "review_status",
}


def freeze_review_packet(
    corpus: CorpusManifest,
    review_path: Path,
    *,
    version: str = "1.0.0",
) -> EvaluationDataset:
    """Convert an entirely approved review packet into a corpus-validated dataset."""

    packet = _load_packet(review_path)
    source = next(
        (item for item in corpus.sources if item.source_document_id == packet["report_id"]),
        None,
    )
    if source is None or source.split is not Split.TEST or packet["split"] != "test":
        raise ValueError("review packet must reference a corpus source frozen in the test split")
    reviewer = _string(packet, "reviewer")
    author = _string(packet, "author")
    if reviewer == author:
        raise ValueError("test reviewer must be independent of the packet author")
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    documents = {document.document_id: document for document in corpus.documents}
    examples: list[GroundedQAExample] = []
    for item in packet["examples"]:
        if not isinstance(item, Mapping):
            raise ValueError("review examples must be objects")
        evidence_key = "quotes" if "quotes" in item else "quote"
        if set(item) != _EXAMPLE_FIELDS | {evidence_key}:
            raise ValueError(f"invalid reviewed example fields: {item.get('example_id')}")
        if item["review_status"] != "approved":
            raise ValueError(f"example is not approved: {item.get('example_id')}")
        chunk_id = _string(item, "chunk_id")
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"review evidence references unknown chunk: {chunk_id}")
        document = documents[chunk.document_id]
        if document.source_sha256 != source.sha256:
            raise ValueError("review evidence crosses the frozen test report boundary")
        if item["page"] != chunk.page_start:
            raise ValueError(f"review page does not match chunk provenance: {item['example_id']}")
        quotes = _reviewed_quotes(item, evidence_key)
        spans = tuple(
            EvidenceSpan(chunk_id, _resolve_layout_normalized_span(chunk.text, quote))
            for quote in quotes
        )
        examples.append(
            GroundedQAExample(
                example_id=_string(item, "example_id"),
                question=_string(item, "question"),
                report_id=source.source_document_id,
                event_id=source.event_id,
                event_family_id=source.event_family_id,
                split=Split.TEST,
                answerable=True,
                reference_answer=_string(item, "reference_answer"),
                rubric=(),
                evidence=spans,
                provenance=ExampleProvenance(
                    author=author,
                    reviewers=(reviewer,),
                    source_document_ids=(source.source_document_id,),
                    is_synthetic=False,
                ),
            )
        )
    dataset = EvaluationDataset(_string(packet, "dataset_id"), version, tuple(examples))
    corpus.validate_dataset(dataset)
    return dataset


def write_frozen_dataset(dataset: EvaluationDataset, path: Path, digest_path: Path) -> str:
    """Write immutable dataset bytes and a companion SHA-256 record."""

    content = (dataset.to_json(indent=2) + "\n").encode()
    digest = hashlib.sha256(content).hexdigest()
    digest_content = f"{digest}  {path.name}\n".encode()
    _require_missing_or_identical(path, content)
    _require_missing_or_identical(digest_path, digest_content)
    if not path.exists():
        atomic_write(path, content)
    if not digest_path.exists():
        atomic_write(digest_path, digest_content)
    return digest


def _load_packet(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("review packet is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != _PACKET_FIELDS:
        raise ValueError("invalid review packet fields")
    if value["schema_version"] != 1 or not isinstance(value["examples"], list):
        raise ValueError("unsupported review packet schema")
    if not value["examples"]:
        raise ValueError("review packet contains no examples")
    return value


def _string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return item


def _reviewed_quotes(item: Mapping[str, Any], evidence_key: str) -> tuple[str, ...]:
    value = item[evidence_key]
    if evidence_key == "quote":
        return (_string(item, "quote"),)
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("quotes must contain at least two reviewed spans")
    if any(not isinstance(quote, str) or not quote.strip() for quote in value):
        raise ValueError("quotes must contain non-empty strings")
    return tuple(value)


def _resolve_layout_normalized_span(text: str, quote: str) -> str:
    compact_text, offsets = _without_whitespace(text)
    compact_quote, _ = _without_whitespace(quote)
    start = compact_text.find(compact_quote)
    if start < 0:
        raise ValueError(f"reviewed quote is not exact after layout normalization: {quote!r}")
    end = start + len(compact_quote)
    return text[offsets[start]:offsets[end - 1] + 1]


def _without_whitespace(value: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(value):
        if not character.isspace():
            characters.append(character)
            offsets.append(offset)
    return "".join(characters), offsets


def _require_missing_or_identical(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite frozen artifact with different bytes: {path}")
