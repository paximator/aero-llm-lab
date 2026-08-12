"""Build evidence-exact retrieval datasets from reviewed declarative specifications."""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample


def build_development_dataset(corpus: CorpusManifest, config_path: Path) -> EvaluationDataset:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    expected = {"dataset_id", "version", "author", "examples"}
    if set(config) != expected or not isinstance(config["examples"], list):
        raise ValueError("invalid retrieval dataset configuration")
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    sources = {source.source_document_id: source for source in corpus.sources}
    documents = {document.document_id: document for document in corpus.documents}
    source_by_digest = {source.sha256: source for source in corpus.sources}
    examples: list[GroundedQAExample] = []
    normalized_questions: set[str] = set()
    for spec in config["examples"]:
        if not isinstance(spec, Mapping):
            raise ValueError("retrieval example specifications must be objects")
        report_id = _required(spec, "report_id")
        source = sources.get(report_id)
        if source is None or source.split is not Split.DEVELOPMENT:
            raise ValueError(f"example source is not a development report: {report_id}")
        question = _required(spec, "question")
        normalized = re.sub(r"\W+", " ", question.casefold()).strip()
        if normalized in normalized_questions:
            raise ValueError(f"duplicate normalized question: {question}")
        normalized_questions.add(normalized)
        chunk_id = spec.get("chunk_id")
        answer = spec.get("answer")
        if chunk_id is None:
            if answer is not None or "quote" in spec:
                raise ValueError("unanswerable specifications cannot contain evidence")
            evidence: tuple[EvidenceSpan, ...] = ()
            rubric = (_required(spec, "rubric"),)
            answerable = False
        else:
            chunk = chunks.get(str(chunk_id))
            if chunk is None:
                raise ValueError(f"unknown evidence chunk: {chunk_id}")
            document = documents[chunk.document_id]
            if source_by_digest[document.source_sha256] != source:
                raise ValueError("evidence chunk crosses report boundary")
            quote = _exact_quote(chunk.text, _required(spec, "quote"))
            if quote is None:
                raise ValueError(f"configured quote is not exact evidence: {spec['id']}")
            evidence = (EvidenceSpan(str(chunk_id), quote),)
            rubric = ()
            answerable = True
        examples.append(
            GroundedQAExample(
                example_id=_required(spec, "id"), question=question,
                report_id=report_id, event_id=source.event_id,
                event_family_id=source.event_family_id, split=source.split,
                answerable=answerable,
                reference_answer=_required(spec, "answer") if answerable else None,
                rubric=rubric, evidence=evidence,
                provenance=ExampleProvenance(
                    author=config["author"], reviewers=(),
                    source_document_ids=(report_id,), is_synthetic=True,
                ),
            )
        )
    dataset = EvaluationDataset(config["dataset_id"], config["version"], tuple(examples))
    corpus.validate_dataset(dataset)
    return dataset


def write_dataset(dataset: EvaluationDataset, path: Path) -> str:
    from aerollm.data.snapshots import atomic_write

    content = (dataset.to_json(indent=2) + "\n").encode()
    atomic_write(path, content)
    return hashlib.sha256(content).hexdigest()


def build_test_review_packet(corpus: CorpusManifest, config_path: Path) -> dict[str, object]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    expected = {"dataset_id", "version", "author", "report_id", "examples"}
    if set(config) != expected or not isinstance(config["examples"], list):
        raise ValueError("invalid test review configuration")
    source = next(
        (item for item in corpus.sources if item.source_document_id == config["report_id"]),
        None,
    )
    if source is None or source.split is not Split.TEST:
        raise ValueError("test review packet must use a frozen test report")
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    documents = {document.document_id: document for document in corpus.documents}
    items = []
    for spec in config["examples"]:
        chunk = chunks.get(_required(spec, "chunk_id"))
        if chunk is None or documents[chunk.document_id].source_sha256 != source.sha256:
            raise ValueError("test draft evidence crosses report boundary")
        configured_quotes = _configured_quotes(spec)
        quotes = [_exact_quote(chunk.text, quote) for quote in configured_quotes]
        if any(quote is None for quote in quotes):
            raise ValueError(f"test draft quote is not exact: {spec['id']}")
        item = {
            "example_id": _required(spec, "id"),
            "question": _required(spec, "question"),
            "reference_answer": _required(spec, "answer"),
            "chunk_id": chunk.chunk_id, "page": chunk.page_start,
            "review_status": "pending-human-review",
        }
        item["quote" if len(quotes) == 1 else "quotes"] = (
            quotes[0] if len(quotes) == 1 else quotes
        )
        items.append(item)
    return {
        "schema_version": 1, "dataset_id": config["dataset_id"],
        "version": config["version"], "author": config["author"],
        "report_id": config["report_id"], "split": "test",
        "reviewer": None, "examples": items,
    }


def _required(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return item


def _configured_quotes(value: Mapping[str, Any]) -> tuple[str, ...]:
    quote = value.get("quote")
    quotes = value.get("quotes")
    if quote is not None and quotes is not None:
        raise ValueError("use either quote or quotes, not both")
    if quote is not None:
        return (_required(value, "quote"),)
    if not isinstance(quotes, list) or len(quotes) < 2:
        raise ValueError("quotes must contain at least two evidence strings")
    if any(not isinstance(item, str) or not item.strip() for item in quotes):
        raise ValueError("quotes must contain non-empty strings")
    return tuple(quotes)


def _exact_quote(text: str, normalized_quote: str) -> str | None:
    pattern = r"\s+".join(re.escape(token) for token in normalized_quote.split())
    match = re.search(pattern, text)
    return match.group(0) if match else None
