import hashlib
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.retrieval_dataset import build_development_dataset


def test_builder_resolves_layout_whitespace_to_exact_quote(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    config = _config(tmp_path, chunk.chunk_id, quote="exact evidence phrase")

    dataset = build_development_dataset(corpus, config)

    assert dataset.examples[0].evidence[0].quote == "exact   evidence\nphrase"
    assert dataset.examples[0].provenance.is_synthetic


def test_builder_rejects_non_exact_evidence(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    config = _config(tmp_path, chunk.chunk_id, quote="invented evidence")

    with pytest.raises(ValueError, match="not exact"):
        build_development_dataset(corpus, config)


def test_builder_rejects_evidence_from_another_report(tmp_path: Path) -> None:
    corpus, chunk = _corpus()
    config = _config(tmp_path, chunk.chunk_id, report_id="other:report")

    with pytest.raises(ValueError, match="not a development report"):
        build_development_dataset(corpus, config)


def _corpus() -> tuple[CorpusManifest, Chunk]:
    text = "An exact   evidence\nphrase appears here."
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        f"doc-{digest}", digest, text, (PageSpan(1, 0, len(text)),), "fixture",
    )
    chunk = Chunk.create(
        document_id=document.document_id, text=text, start=0, end=len(text),
        page_start=1, page_end=1,
    )
    source = SourceManifestEntry(
        "case:report", "case", "case", Split.DEVELOPMENT, digest,
    )
    return CorpusManifest("v1", (source,), (document,), (chunk,)), chunk


def _config(
    tmp_path: Path, chunk_id: str, *, quote: str = "exact evidence phrase",
    report_id: str = "case:report",
) -> Path:
    path = tmp_path / "dataset.toml"
    path.write_text(
        "\n".join(
            (
                'dataset_id = "fixture"', 'version = "1"', 'author = "author"',
                "[[examples]]", 'id = "question-1"', f'report_id = "{report_id}"',
                'question = "What appears?"', 'answer = "Evidence appears."',
                f'chunk_id = "{chunk_id}"', f'quote = "{quote}"',
            )
        ),
        encoding="utf-8",
    )
    return path

