"""Build the canonical three-report evaluation-suite-v2 development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from aerollm.common.schemas import SourceDocument, Split
from aerollm.data.chunking import ChunkingConfig, chunk_document
from aerollm.data.pdf import parse_pdf_snapshot
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.schemas import EvaluationDataset

_REPORTS = {
    "CEN16MA036": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1603.pdf",
    "DCA13MA081": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1501.pdf",
    "ERA14MA271": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1503.pdf",
}


def build(pdf_root: Path, chunking_config: Path) -> CorpusManifest:
    config = ChunkingConfig.from_dict(tomllib.loads(chunking_config.read_text(encoding="utf-8")))
    sources = []
    documents = []
    chunks = []
    for event_id, url in _REPORTS.items():
        path = pdf_root / f"{event_id}.pdf"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = SourceDocument(
            source="ntsb-report",
            source_id=f"{event_id}:report",
            source_url=url,
            publisher="National Transportation Safety Board",
            retrieved_at=datetime.now(UTC),
            sha256=digest,
            artifact_path=str(path.resolve()),
            event_id=event_id,
        )
        document = parse_pdf_snapshot(source)
        manifest = chunk_document(document, config)
        sources.append(
            SourceManifestEntry(source.source_id, event_id, event_id, Split.DEVELOPMENT, digest)
        )
        documents.append(document)
        chunks.extend(manifest.chunks)
    return CorpusManifest(
        "evaluation-suite-v2-development-v1", tuple(sources), tuple(documents), tuple(chunks)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_root", type=Path)
    parser.add_argument("--chunking-config", type=Path, default=Path("configs/data/chunking.toml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/v2/development.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = build(args.pdf_root, args.chunking_config)
    dataset = EvaluationDataset.from_json(args.dataset.read_text(encoding="utf-8"))
    corpus.validate_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(corpus.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(corpus.documents)} documents and {len(corpus.chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
