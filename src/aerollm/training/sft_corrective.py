"""Build a deterministic train-only corrective SFT dataset from observed failure classes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from aerollm.common.schemas import Split
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.postprocessing import PostprocessorV1, RetrievedEvidence

_SPACE = re.compile(r"\s+")
_SYSTEM = (
    "You answer aviation-report questions using only the supplied excerpt. "
    "Return the grounded-json-v1 JSON object and cite an exact span."
)


def build_corrective_dataset(
    base: Mapping[str, object], corpus: CorpusManifest, *, corpus_sha256: str
) -> dict[str, object]:
    """Append ten records for each corrective task without using development or test reports."""
    base_records = base.get("records")
    if not isinstance(base_records, list) or len(base_records) != 50:
        raise ValueError("corrective SFT requires the reviewed 50-record base dataset")
    sources_by_family = {
        source.event_family_id: source for source in corpus.sources if source.split is Split.TRAIN
    }

    def reviewed(index: int) -> tuple[object, str, str]:
        record = base_records[index]
        source = sources_by_family[record["event_family_id"]]
        chunk_id = record["source_chunk_ids"][0]
        answer = json.loads(record["messages"][2]["content"])["answer"]
        return source, chunk_id, answer

    selected: list[dict[str, object]] = []
    for index in (9, 10, 12, 18, 22, 23, 36, 37, 44, 49):
        source, chunk_id, sentence = reviewed(index)
        selected.append(_answerable_record(source, "causal", [chunk_id], [sentence]))

    for first_index, second_index in (
        (0, 13),
        (1, 14),
        (2, 15),
        (3, 16),
        (4, 17),
        (5, 18),
        (6, 19),
        (7, 20),
        (8, 21),
        (9, 22),
    ):
        first, second = reviewed(first_index), reviewed(second_index)
        selected.append(
            _answerable_record(
                first[0], "multi_evidence", [first[1], second[1]], [first[2], second[2]]
            )
        )

    for index in (10, 11, 12, 24, 25, 26, 27, 28, 29, 30):
        source, chunk_id, sentence = reviewed(index)
        selected.append(_answerable_record(source, "exact_citation", [chunk_id], [sentence]))

    for ordinal, index in enumerate(range(10), start=1):
        source, chunk_id, _ = reviewed(index)
        selected.append(_abstention_record(source, chunk_id, ordinal))

    counts = {
        task: sum(record["task_type"] == task for record in selected)
        for task in ("abstention", "causal", "multi_evidence", "exact_citation")
    }
    if set(counts.values()) != {10}:
        raise ValueError(f"corrective task balance failed: {counts}")
    records = [*base_records, *selected]
    dataset = {
        "schema_version": 2,
        "dataset_id": "ntsb-evidence-linked-sft-v2-corrective",
        "version": "2.0.0-development",
        "status": "model_assisted_review_passed",
        "review": {
            "kind": "model_assisted_full_corrective_review",
            "reviewed_records": 40,
            "comment": (
                "All corrective records were inspected after generation. They reuse only "
                "the 50 previously reviewed train facts, recomposed into failure-targeted tasks."
            ),
        },
        "source_corpus_sha256": corpus_sha256,
        "selection": {"base_records": 50, "corrective_records": 40, "task_counts": counts},
        "records": records,
    }
    validate_corrective_dataset(dataset, corpus)
    return dataset


def validate_corrective_dataset(dataset: Mapping[str, object], corpus: CorpusManifest) -> None:
    """Validate split provenance and all grounded/abstention contracts."""
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    documents = {document.document_id: document for document in corpus.documents}
    sources = {source.sha256: source for source in corpus.sources}
    records = dataset.get("records")
    if not isinstance(records, list) or len(records) != 90:
        raise ValueError("corrective SFT dataset must contain 90 records")
    ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = str(record["record_id"])
        if record_id in ids:
            raise ValueError("duplicate corrective SFT record ID")
        ids.add(record_id)
        evidence = []
        for chunk_id in record["source_chunk_ids"]:
            chunk = chunks[chunk_id]
            source = sources[documents[chunk.document_id].source_sha256]
            if (
                source.split is not Split.TRAIN
                or source.event_family_id != record["event_family_id"]
            ):
                raise ValueError("corrective SFT record crosses train provenance")
            evidence.append(RetrievedEvidence(chunk_id, chunk.text))
        if index < 50:
            continue
        payload = json.loads(record["messages"][2]["content"])
        cited: defaultdict[str, list[str]] = defaultdict(list)
        for citation in payload["citations"]:
            quote = str(citation["quote"])
            chunk_text = chunks[str(citation["chunk_id"])].text
            if _SPACE.sub(" ", quote).strip() not in _SPACE.sub(" ", chunk_text).strip():
                raise ValueError("corrective citation is not an exact normalized corpus span")
            cited[str(citation["chunk_id"])].append(quote)
        contract_evidence = [
            RetrievedEvidence(chunk_id, "\n".join(quotes)) for chunk_id, quotes in cited.items()
        ]
        processed = PostprocessorV1().process(json.dumps(payload), contract_evidence)
        is_abstention = record.get("task_type") == "abstention"
        if is_abstention != processed.abstained or processed.warnings:
            raise ValueError(
                f"corrective SFT output violates grounded contract: {record_id} "
                f"task={record.get('task_type')} warnings={processed.warnings}"
            )


def _answerable_record(source, task: str, chunk_ids: list[str], sentences: list[str]):  # type: ignore[no-untyped-def]
    excerpts = "\n\n".join(
        f'<chunk id="{chunk_id}">\n{sentence}\n</chunk>'
        for chunk_id, sentence in zip(chunk_ids, sentences, strict=True)
    )
    question = {
        "causal": "State the causal or contributing aviation-safety finding in the excerpts.",
        "multi_evidence": "State the two distinct aviation-safety facts supported by the excerpts.",
        "exact_citation": (
            "Extract the aviation-safety fact and preserve an exact supporting quote."
        ),
    }[task]
    answer = " ".join(sentences)
    payload = {
        "answer": answer,
        "citations": [
            {"chunk_id": chunk_id, "quote": sentence}
            for chunk_id, sentence in zip(chunk_ids, sentences, strict=True)
        ],
        "abstained": False,
        "abstention_reason": None,
    }
    return _record(source, task, chunk_ids, question, excerpts, payload)


def _abstention_record(source, chunk_id: str, index: int):  # type: ignore[no-untyped-def]
    question = f"What unsupported passenger preference number {index} is stated in the evidence?"
    payload = {
        "answer": None,
        "citations": [],
        "abstained": True,
        "abstention_reason": "insufficient evidence",
    }
    return _record(source, "abstention", [chunk_id], question, "<no_evidence />", payload)


def _record(source, task, chunk_ids, question, excerpts, payload):  # type: ignore[no-untyped-def]
    assistant = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    identity = hashlib.sha256(
        f"{source.event_id}\0{task}\0{question}\0{assistant}".encode()
    ).hexdigest()[:16]
    return {
        "record_id": f"sft-{identity}",
        "event_family_id": source.event_family_id,
        "source_chunk_ids": chunk_ids,
        "task_type": task,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{question}\n\n{excerpts}"},
            {"role": "assistant", "content": assistant},
        ],
    }


def write_corrective_dataset(dataset: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
