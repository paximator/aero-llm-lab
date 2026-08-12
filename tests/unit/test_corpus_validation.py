import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk, ContentKind
from aerollm.common.schemas import Document, EvidenceSpan, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample
from aerollm.evaluation.validate_cli import main

FIXTURES = Path(__file__).parents[1] / "fixtures" / "corpus"


def records(
    *, kind: ContentKind = ContentKind.TEXT, split: Split = Split.TEST
) -> tuple[SourceManifestEntry, Document, Chunk]:
    text = (FIXTURES / f"{kind.value}.txt").read_text(encoding="utf-8")
    source = SourceManifestEntry(
        source_document_id=f"source-{kind.value}",
        event_id=f"event-{kind.value}",
        event_family_id=f"family-{kind.value}",
        split=split,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    document = Document(
        document_id=f"doc-{source.sha256}",
        source_sha256=source.sha256,
        text=text,
        pages=(PageSpan(page_number=1, start_offset=0, end_offset=len(text)),),
        parser="fixture-parser-v1",
    )
    chunk = Chunk.create(
        document_id=document.document_id,
        text=text,
        start=0,
        end=len(text),
        page_start=1,
        page_end=1,
        section="fixture",
        kind=kind,
    )
    return source, document, chunk


def dataset_for(source: SourceManifestEntry, chunk: Chunk) -> EvaluationDataset:
    quote = chunk.text.splitlines()[-1]
    example = GroundedQAExample(
        example_id=f"qa-{chunk.kind.value}",
        question="What does the report state?",
        report_id=source.source_document_id,
        event_id=source.event_id,
        event_family_id=source.event_family_id,
        split=source.split,
        answerable=True,
        reference_answer=quote,
        rubric=("Uses the stated value.",),
        evidence=(EvidenceSpan(chunk.chunk_id, quote),),
        provenance=ExampleProvenance(
            author="analyst-a",
            reviewers=("analyst-b",) if source.split is Split.TEST else (),
            source_document_ids=(source.source_document_id,),
        ),
    )
    return EvaluationDataset("corpus-fixture", "1.0.0", (example,))


@pytest.mark.parametrize("kind", list(ContentKind))
def test_text_table_and_ocr_chunks_validate(kind: ContentKind) -> None:
    source, document, chunk = records(kind=kind)
    corpus = CorpusManifest("1.0.0", (source,), (document,), (chunk,))

    corpus.validate_dataset(dataset_for(source, chunk))


def test_document_and_chunk_ids_are_content_derived() -> None:
    _, document, chunk = records()
    document_data, chunk_data = document.to_dict(), chunk.to_dict()
    document_data["source_sha256"] = "0" * 64
    chunk_data["text"] = "tampered"

    with pytest.raises(ValueError, match="document_id"):
        Document.from_dict(document_data)
    with pytest.raises(ValueError, match="chunk_id"):
        Chunk.from_dict(chunk_data)


def test_chunk_must_match_document_offsets() -> None:
    source, document, chunk = records()
    shifted = Chunk.create(
        document_id=document.document_id,
        text=chunk.text,
        start=1,
        end=len(chunk.text) + 1,
        page_start=1,
        page_end=1,
    )

    with pytest.raises(ValueError, match="does not match document offsets"):
        CorpusManifest("1", (source,), (document,), (shifted,))


def test_event_family_cannot_cross_source_splits() -> None:
    first, document, chunk = records(split=Split.TRAIN)
    second = SourceManifestEntry(
        "source-other", "event-other", first.event_family_id, Split.TEST, "0" * 64
    )

    with pytest.raises(ValueError, match="crosses source splits"):
        CorpusManifest("1", (first, second), (document,), (chunk,))


def test_gold_quote_must_be_exact_span_in_declared_source() -> None:
    source, document, chunk = records()
    corpus = CorpusManifest("1", (source,), (document,), (chunk,))
    dataset = dataset_for(source, chunk)
    original = dataset.examples[0]
    invalid = GroundedQAExample(
        example_id=original.example_id,
        question=original.question,
        report_id=original.report_id,
        event_id=original.event_id,
        event_family_id=original.event_family_id,
        split=original.split,
        answerable=True,
        reference_answer=original.reference_answer,
        rubric=original.rubric,
        evidence=(EvidenceSpan(chunk.chunk_id, "not in the chunk"),),
        provenance=original.provenance,
    )

    with pytest.raises(ValueError, match="not an exact span"):
        corpus.validate_dataset(EvaluationDataset("fixture", "1", (invalid,)))


def test_corpus_json_round_trip_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source, document, chunk = records()
    corpus = CorpusManifest("1.0.0", (source,), (document,), (chunk,))
    corpus_json = json.dumps(corpus.to_dict(), sort_keys=True)
    assert CorpusManifest.from_json(corpus_json) == corpus
    corpus_path, dataset_path = tmp_path / "corpus.json", tmp_path / "dataset.json"
    corpus_path.write_text(corpus_json, encoding="utf-8")
    dataset_path.write_text(dataset_for(source, chunk).to_json(), encoding="utf-8")

    assert main([str(corpus_path), str(dataset_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["chunks"] == 1
