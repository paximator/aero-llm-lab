"""Build the mixed draft/blank human-review workbook for evaluation suite v2."""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry


def build_suite_v2_review(
    corpus: CorpusManifest, proposal_path: Path, development_draft_path: Path,
    frozen_selection_path: Path,
) -> dict[str, object]:
    proposal = tomllib.loads(proposal_path.read_text(encoding="utf-8"))
    draft = tomllib.loads(development_draft_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_selection_path.read_text(encoding="utf-8"))
    report_urls = {
        item["event_id"]: item["report_url"] for item in frozen["event_families"]
    }
    targets = {
        item["task_type"]: int(item["target"])
        for item in proposal["coverage"]["primary_task_types"]
    }
    sources = {source.event_id: source for source in corpus.sources}
    chunks_by_source = _chunks_by_source(corpus)
    examples = [
        _development_item(spec, draft, sources, chunks_by_source, report_urls)
        for spec in draft["examples"]
    ]
    development_counts = Counter(item["task_type"] for item in examples)
    remaining = {
        task: target - development_counts.get(task, 0)
        for task, target in targets.items()
    }
    if any(count < 0 for count in remaining.values()):
        raise ValueError("development draft exceeds an approved task quota")
    test_sources = sorted(
        (source for source in corpus.sources if source.split.value == "test"),
        key=lambda source: source.event_id,
    )
    test_tasks = [task for task, count in remaining.items() for _ in range(count)]
    expected_test = len(test_sources) * 9
    if len(test_tasks) != expected_test:
        raise ValueError("remaining quotas do not provide nine slots per test family")
    for index, task in enumerate(test_tasks):
        source = test_sources[index % len(test_sources)]
        examples.append(_blank_test_item(source, task, index + 1, report_urls))
    counts = Counter(item["task_type"] for item in examples)
    if counts != Counter(targets) or len(examples) != proposal["target_total_examples"]:
        raise ValueError("review workbook does not satisfy approved coverage quotas")
    family_counts = Counter(item["event_family_id"] for item in examples)
    family_cap = proposal["coverage"]["maximum_examples_per_event_family"]
    if any(count > family_cap for count in family_counts.values()):
        raise ValueError("review workbook exceeds the per-family cap")
    return {
        "schema_version": 1,
        "dataset_id": draft["dataset_id"],
        "version": draft["version"],
        "status": "human_review_and_test_authoring_required",
        "corpus_version": corpus.version,
        "coverage": {
            "total": len(examples),
            "by_task_type": dict(sorted(counts.items())),
            "by_split": dict(sorted(Counter(item["split"] for item in examples).items())),
            "by_event_family": dict(sorted(family_counts.items())),
        },
        "instructions": (
            "Review and correct development drafts. Test slots must be independently "
            "human-authored from their assigned report before they can be approved."
        ),
        "examples": examples,
    }


def write_suite_v2_review(packet: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _development_item(
    spec: Mapping[str, Any], draft: Mapping[str, Any],
    sources: Mapping[str, SourceManifestEntry],
    chunks_by_source: Mapping[str, list[Mapping[str, Any]]],
    report_urls: Mapping[str, str],
) -> dict[str, object]:
    source = sources[str(spec["event_id"])]
    if source.split.value != "development":
        raise ValueError(f"development draft crosses split: {source.event_id}")
    quotes = spec.get("quotes", [])
    evidence = [
        _resolve_quote(chunks_by_source[source.source_document_id], str(quote))
        for quote in quotes
    ]
    answerable = bool(evidence)
    task = str(spec["task_type"])
    tags = ["natural_language", "multi_evidence" if len(evidence) > 1 else "single_evidence"]
    if task in {"numeric", "date_time", "entity_identifier"}:
        tags.append("identifier_heavy")
    return {
        "example_id": spec["id"],
        "question": spec["question"],
        "report_id": source.source_document_id,
        "report_url": report_urls[source.event_id],
        "event_id": source.event_id,
        "event_family_id": source.event_family_id,
        "split": source.split.value,
        "task_type": task,
        "scoring_strategy": _scoring_strategy(task),
        "secondary_tags": tags,
        "answerable": answerable,
        "reference_answer": spec.get("answer"),
        "required_key_facts": [],
        "structured_target": None,
        "evidence": evidence,
        "unanswerable_search_note": spec.get("rubric"),
        "citation_required": answerable,
        "author": draft["author"],
        "reviewers": [],
        "review_status": "draft",
        "is_synthetic": True,
        "review_notes": "",
    }


def _blank_test_item(
    source: SourceManifestEntry, task: str, ordinal: int,
    report_urls: Mapping[str, str],
) -> dict[str, object]:
    return {
        "example_id": f"test-slot-{ordinal:02d}-{source.event_id.casefold()}",
        "question": "",
        "report_id": source.source_document_id,
        "report_url": report_urls[source.event_id],
        "event_id": source.event_id,
        "event_family_id": source.event_family_id,
        "split": "test",
        "task_type": task,
        "scoring_strategy": _scoring_strategy(task),
        "secondary_tags": [],
        "answerable": task != "unanswerable",
        "reference_answer": None,
        "required_key_facts": [],
        "structured_target": None,
        "evidence": [],
        "unanswerable_search_note": None,
        "citation_required": task != "unanswerable",
        "author": None,
        "reviewers": [],
        "review_status": "authoring_required",
        "is_synthetic": False,
        "review_notes": "",
    }


def _chunks_by_source(corpus: CorpusManifest) -> dict[str, list[Mapping[str, Any]]]:
    source_by_digest = {source.sha256: source for source in corpus.sources}
    document_source = {
        document.document_id: source_by_digest[document.source_sha256]
        for document in corpus.documents
    }
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in corpus.chunks:
        source = document_source[chunk.document_id]
        result[source.source_document_id].append(
            {
                "chunk_id": chunk.chunk_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text,
            }
        )
    return result


def _resolve_quote(chunks: list[Mapping[str, Any]], normalized_quote: str) -> dict[str, object]:
    pattern = re.compile(r"\s+".join(re.escape(token) for token in normalized_quote.split()))
    for chunk in chunks:
        match = pattern.search(str(chunk["text"]))
        if match:
            return {
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "quote": match.group(0),
            }
    raise ValueError(f"draft quote is not an exact corpus span: {normalized_quote}")


def _scoring_strategy(task: str) -> str:
    return {
        "numeric": "numeric_exact_or_tolerance_v1",
        "date_time": "normalized_datetime_v1",
        "entity_identifier": "normalized_entity_set_v1",
        "categorical_yes_no": "categorical_exact_v1",
        "causal_finding": "required_key_facts_v1",
        "multi_evidence_grounded_qa": "required_key_facts_v1",
        "unanswerable": "abstention_v1",
    }[task]
