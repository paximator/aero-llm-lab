import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk, ContentKind, Document, PageSpan
from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample
from aerollm.retrieval.bm25 import BM25Index, IndexManifest, tokenize
from aerollm.retrieval.cli import main
from aerollm.retrieval.evaluation import evaluate_retrieval


def corpus_and_dataset(*, split: Split = Split.TEST) -> tuple[CorpusManifest, EvaluationDataset]:
    records = (
        ("text", ContentKind.TEXT, "The event occurred during the landing phase."),
        ("table", ContentKind.TABLE, "Case CEN24LA001 | Injury level: None"),
        ("ocr", ContentKind.OCR, "WEA7HER observation reported VMC conditions."),
        ("distractor", ContentKind.TEXT, "The maintenance record lists an engine inspection."),
    )
    sources: list[SourceManifestEntry] = []
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for name, kind, text in records:
        source = SourceManifestEntry(
            f"source-{name}",
            f"event-{name}",
            f"family-{name}",
            split,
            hashlib.sha256(text.encode()).hexdigest(),
        )
        document = Document.create(
            source_document_id=source.source_document_id,
            event_id=source.event_id,
            event_family_id=source.event_family_id,
            split=split,
            parser_version="fixture-v1",
            text=text,
            pages=(PageSpan(1, 0, len(text)),),
        )
        chunk = Chunk.create(
            document_id=document.document_id,
            text=text,
            start=0,
            end=len(text),
            page_start=1,
            page_end=1,
            section=name,
            kind=kind,
        )
        sources.append(source)
        documents.append(document)
        chunks.append(chunk)

    questions = (
        ("landing", "During which phase did the event occur?", "landing phase"),
        ("table", "What injury level is listed for CEN24LA001?", "Injury level: None"),
        ("ocr", "What conditions did the WEA7HER observation report?", "VMC conditions"),
    )
    by_name = {chunk.section: chunk for chunk in chunks}
    source_by_name = {
        source.source_document_id.removeprefix("source-"): source for source in sources
    }
    examples: list[GroundedQAExample] = []
    for name, question, quote in questions:
        source = source_by_name["text" if name == "landing" else name]
        chunk = by_name["text" if name == "landing" else name]
        examples.append(_example(source, question, quote, chunk, split))
    unanswerable_source = source_by_name["distractor"]
    examples.append(
        GroundedQAExample(
            example_id="qa-unanswerable",
            question="What was the pilot's motive?",
            report_id=unanswerable_source.source_document_id,
            event_id=unanswerable_source.event_id,
            event_family_id=unanswerable_source.event_family_id,
            split=split,
            answerable=False,
            reference_answer=None,
            rubric=("Abstains; motive is absent.",),
            evidence=(),
            provenance=_provenance(unanswerable_source, split),
        )
    )
    return (
        CorpusManifest("retrieval-fixture-v1", tuple(sources), tuple(documents), tuple(chunks)),
        EvaluationDataset("retrieval-fixture", "1.0.0", tuple(examples)),
    )


def _provenance(source: SourceManifestEntry, split: Split) -> ExampleProvenance:
    return ExampleProvenance(
        "analyst-a",
        ("analyst-b",) if split is Split.TEST else (),
        (source.source_document_id,),
    )


def _example(
    source: SourceManifestEntry,
    question: str,
    quote: str,
    chunk: Chunk,
    split: Split,
) -> GroundedQAExample:
    return GroundedQAExample(
        example_id=f"qa-{chunk.section}",
        question=question,
        report_id=source.source_document_id,
        event_id=source.event_id,
        event_family_id=source.event_family_id,
        split=split,
        answerable=True,
        reference_answer=quote,
        rubric=("Finds the stated value.",),
        evidence=(EvidenceSpan(chunk.chunk_id, quote),),
        provenance=_provenance(source, split),
    )


def test_tokenizer_preserves_aviation_identifiers() -> None:
    assert tokenize("Case CEN24LA001 / RWY-27") == ("case", "cen24la001", "rwy-27")


def test_index_identity_is_stable_under_input_order() -> None:
    corpus, _ = corpus_and_dataset()
    digest = "a" * 64

    first = BM25Index(corpus.chunks, corpus_sha256=digest)
    second = BM25Index(tuple(reversed(corpus.chunks)), corpus_sha256=digest)

    assert first.manifest == second.manifest
    assert IndexManifest.from_dict(first.manifest.to_dict()) == first.manifest
    assert (
        first.search("CEN24LA001", k=1).hits[0].chunk_id
        == second.search("CEN24LA001", k=1).hits[0].chunk_id
    )


def test_retrieval_metrics_and_slices_are_reported() -> None:
    corpus, dataset = corpus_and_dataset()
    index = BM25Index(corpus.chunks, corpus_sha256="b" * 64)

    report = evaluate_retrieval(dataset, index, corpus.chunks, k=2)

    assert report.aggregate.scored_count == 3
    assert report.aggregate.recall_at_k == 1.0
    assert report.aggregate.mrr == 1.0
    assert report.aggregate.ndcg_at_k == 1.0
    assert report.slices["answerable:false"].scored_count == 0
    assert report.slices["answerable:false"].mrr is None
    assert {"content:text", "content:table", "content:ocr", "query:identifier"} <= set(
        report.slices
    )


def test_lexical_baseline_exposes_paraphrase_failure_with_distractor() -> None:
    corpus, _ = corpus_and_dataset()
    index = BM25Index(corpus.chunks, corpus_sha256="d" * 64)
    landing_chunk = next(chunk for chunk in corpus.chunks if chunk.section == "text")

    result = index.search("Was touchdown completed after the engine inspection?", k=1)

    assert result.hits[0].chunk_id != landing_chunk.chunk_id


def test_training_examples_are_rejected_from_retrieval_evaluation() -> None:
    corpus, dataset = corpus_and_dataset(split=Split.TRAIN)
    index = BM25Index(corpus.chunks, corpus_sha256="c" * 64)

    with pytest.raises(ValueError, match="development or frozen test"):
        evaluate_retrieval(dataset, index, corpus.chunks)


def test_cli_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    corpus, dataset = corpus_and_dataset()
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    json_output = tmp_path / "retrieval.json"
    markdown_output = tmp_path / "retrieval.md"
    corpus_path.write_text(json.dumps(corpus.to_dict()), encoding="utf-8")
    dataset_path.write_text(dataset.to_json(), encoding="utf-8")

    result = main(
        [
            str(corpus_path),
            str(dataset_path),
            "--k",
            "2",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["evaluator"] == "deterministic-retrieval-v1"
    assert (
        payload["index_manifest"]["corpus_sha256"]
        == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    )
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "Recall@k" in markdown
    assert "answerable:false" in markdown
