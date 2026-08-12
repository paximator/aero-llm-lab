import hashlib
import json
from pathlib import Path

import pytest

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Document, PageSpan, Split
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.reporting.data_pipeline import collect_metrics, render_markdown, write_results


def test_metrics_are_derived_from_verified_artifacts(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)

    metrics = collect_metrics(*paths, code_revision="abc123")
    metrics_path, report_path = write_results(metrics, tmp_path / "results")

    assert metrics["source_availability_percent"] == 50.0
    assert metrics["pages"] == 1
    assert metrics["chunks"] == 1
    assert json.loads(metrics_path.read_text())["corpus_sha256"] == metrics["corpus_sha256"]
    report = report_path.read_text()
    assert "50.0%" in report
    assert "abc123" in report


def test_changed_corpus_is_rejected(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    paths[0].write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        collect_metrics(*paths, code_revision="abc123")


def test_report_contains_required_sections(tmp_path: Path) -> None:
    metrics = collect_metrics(*_artifacts(tmp_path), code_revision="abc123")
    report = render_markdown(metrics)

    for heading in (
        "## Objective", "## Results", "## Source failures", "## Reproduction",
        "## Provenance", "## Limitations and decision",
    ):
        assert heading in report


def _artifacts(tmp_path: Path) -> list[Path]:
    text = "A report page with enough text for one exact evidence chunk."
    source_digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        f"doc-{source_digest}", source_digest, text,
        (PageSpan(1, 0, len(text)),), "fixture-parser",
    )
    chunk = Chunk.create(
        document_id=document.document_id, text=text, start=0, end=len(text),
        page_start=1, page_end=1,
    )
    source = SourceManifestEntry("case:report", "case", "case", Split.TRAIN, source_digest)
    corpus = CorpusManifest("pilot-v1", (source,), (document,), (chunk,))
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    build_path = tmp_path / "build.json"
    build_path.write_text(
        json.dumps(
            {
                "corpus_sha256": digest, "config_fingerprint": "sha256:fixture",
                "validation": {"valid": True, "errors": [], "warnings": []},
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"candidates": [{}, {}]}), encoding="utf-8")
    failures_path = tmp_path / "failures.json"
    failures_path.write_text(
        json.dumps({"failed": [{"error": "unavailable", "source_id": "other"}]}),
        encoding="utf-8",
    )
    return [corpus_path, build_path, plan_path, failures_path]

