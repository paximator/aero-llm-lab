import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.data.chunking import ChunkingConfig, ChunkManifest
from aerollm.data.corpus import CorpusConfig, build_corpus, write_corpus
from aerollm.data.splits import assign_family_splits
from aerollm.data.validation import validate_built_corpus
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry


def test_split_assignment_is_stable_balanced_and_family_level() -> None:
    families = [f"event-{number:02d}" for number in range(20)]
    fractions = {Split.TRAIN: 0.8, Split.DEVELOPMENT: 0.1, Split.TEST: 0.1}

    first = assign_family_splits(families, seed="fixed", fractions=fractions)
    second = assign_family_splits(reversed(families), seed="fixed", fractions=fractions)

    assert first == second
    assert list(first.values()).count(Split.TRAIN) == 16
    assert list(first.values()).count(Split.DEVELOPMENT) == 2
    assert list(first.values()).count(Split.TEST) == 2


def test_split_assignment_rejects_invalid_fractions() -> None:
    fractions = {Split.TRAIN: 0.8, Split.DEVELOPMENT: 0.2, Split.TEST: 0.2}
    with pytest.raises(ValueError, match="sum to 1"):
        assign_family_splits(["event"], seed="fixed", fractions=fractions)


def test_builder_verifies_inputs_and_is_byte_deterministic(tmp_path: Path) -> None:
    source_manifest, config = _input_report(tmp_path, event_id="DCA20MA059")

    first = build_corpus([source_manifest], config)
    second = build_corpus([source_manifest], config)
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first_digest, first_build = write_corpus(first, first_path)
    second_digest, second_build = write_corpus(second, second_path)

    assert first.corpus == second.corpus
    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_build.exists() and second_build.exists()
    assert first.corpus.sources[0].split is Split.TRAIN
    assert first.validation.valid


def test_builder_rejects_changed_parsed_artifact(tmp_path: Path) -> None:
    source_manifest, config = _input_report(tmp_path, event_id="DCA20MA059")
    value = json.loads(source_manifest.read_text(encoding="utf-8"))
    parsed = Path(value["derived_artifacts"][0]["artifact_path"])
    parsed.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="parsed artifact digest mismatch"):
        build_corpus([source_manifest], config)


def test_builder_honors_explicit_frozen_split(tmp_path: Path) -> None:
    source_manifest, config = _input_report(tmp_path, event_id="DCA20MA059")

    result = build_corpus(
        [source_manifest], config,
        explicit_family_splits={"DCA20MA059": Split.TEST},
    )

    assert result.corpus.sources[0].split is Split.TEST


def test_builder_rejects_incomplete_explicit_split_map(tmp_path: Path) -> None:
    source_manifest, config = _input_report(tmp_path, event_id="DCA20MA059")

    with pytest.raises(ValueError, match="explicit family split mismatch"):
        build_corpus([source_manifest], config, explicit_family_splits={})


def test_validation_reports_uncovered_nonempty_pages() -> None:
    text = "first page\nsecond page"
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        f"doc-{digest}", digest, text,
        (PageSpan(1, 0, 10), PageSpan(2, 10, len(text))), "test-parser",
    )
    chunk = Chunk.create(
        document_id=document.document_id, text=text[:10], start=0, end=10,
        page_start=1, page_end=1,
    )
    source = SourceManifestEntry("event:report", "event", "event", Split.TRAIN, digest)
    corpus = CorpusManifest("v1", (source,), (document,), (chunk,))

    report = validate_built_corpus(corpus, minimum_chunk_characters=1)

    assert not report.valid
    assert "page 2" in report.errors[0]


def _input_report(tmp_path: Path, *, event_id: str) -> tuple[Path, CorpusConfig]:
    raw = b"%PDF synthetic report"
    source_digest = hashlib.sha256(raw).hexdigest()
    text = "PROBABLE CAUSE\n\nThe probable cause was loss of control."
    document = Document(
        f"doc-{source_digest}", source_digest, text,
        (PageSpan(1, 0, len(text)),), "test-parser",
    )
    parsed_path = tmp_path / "parsed.json"
    parsed_bytes = (json.dumps(document.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    parsed_path.write_bytes(parsed_bytes)
    parsed_digest = hashlib.sha256(parsed_bytes).hexdigest()
    chunk = Chunk.create(
        document_id=document.document_id, text=text, start=0, end=len(text),
        page_start=1, page_end=1, section="PROBABLE CAUSE",
    )
    chunk_root = tmp_path / "chunks"
    chunk_root.mkdir()
    ChunkManifest(document.document_id, source_digest, ChunkingConfig(), (chunk,)).write(
        chunk_root / f"{document.document_id}.json"
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2, "source": "ntsb", "created_at": "2026-01-01T00:00:00Z",
                "snapshots": [
                    {
                        "source_id": f"{event_id}:report", "source_url": "https://ntsb.test/report.pdf",
                        "sha256": source_digest, "artifact_path": "ignored.pdf",
                        "retrieved_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "derived_artifacts": [
                    {
                        "artifact_id": document.document_id, "kind": "parsed-document",
                        "sha256": parsed_digest, "artifact_path": parsed_path.as_posix(),
                    }
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    config = CorpusConfig(
        "pilot-v1", "fixed", 0.8, 0.1, 0.1, 20,
        chunk_root, tmp_path / "corpus.json",
    )
    return source_manifest, config
