import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.retrieval_freeze import freeze_review_packet, write_frozen_dataset


def test_freeze_resolves_reviewed_layout_and_multiple_spans(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    review = _review(tmp_path, chunk.chunk_id)

    dataset = freeze_review_packet(corpus, review)

    example = dataset.examples[0]
    assert example.split is Split.TEST
    assert example.provenance.reviewers == ("reviewer",)
    assert [span.quote for span in example.evidence] == [
        "radio                               altitude was 2           66 ft",
        "six\nseconds before collision",
    ]
    corpus.validate_dataset(dataset)


def test_freeze_rejects_unapproved_example(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    review = _review(tmp_path, chunk.chunk_id, status="pending-human-review")

    with pytest.raises(ValueError, match="not approved"):
        freeze_review_packet(corpus, review)


def test_frozen_artifact_cannot_be_silently_changed(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    dataset = freeze_review_packet(corpus, _review(tmp_path, chunk.chunk_id))
    output = tmp_path / "test.json"
    digest_output = tmp_path / "test.sha256"

    digest = write_frozen_dataset(dataset, output, digest_output)
    assert digest_output.read_text() == f"{digest}  test.json\n"
    write_frozen_dataset(dataset, output, digest_output)
    output.write_text("different", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_frozen_dataset(dataset, output, digest_output)


def _corpus() -> tuple[CorpusManifest, Chunk]:
    text = (
        "The radio                               altitude was 2           66 ft. "
        "It happened six\nseconds before collision."
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        f"doc-{digest}", digest, text, (PageSpan(7, 0, len(text)),), "fixture",
    )
    chunk = Chunk.create(
        document_id=document.document_id, text=text, start=0, end=len(text),
        page_start=7, page_end=7,
    )
    source = SourceManifestEntry("case:report", "case", "family", Split.TEST, digest)
    return CorpusManifest("v1", (source,), (document,), (chunk,)), chunk


def _review(tmp_path: Path, chunk_id: str, *, status: str = "approved") -> Path:
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "fixture-test",
                "version": "1.0.0-draft",
                "author": "author",
                "report_id": "case:report",
                "split": "test",
                "reviewer": "reviewer",
                "examples": [
                    {
                        "example_id": "altitude",
                        "question": "What altitude and timing were recorded?",
                        "reference_answer": "266 ft, six seconds before collision.",
                        "chunk_id": chunk_id,
                        "page": 7,
                        "quotes": [
                            "radio altitude was 266 ft",
                            "six seconds before collision",
                        ],
                        "review_status": status,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
