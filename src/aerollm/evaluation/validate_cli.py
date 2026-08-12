"""Validate a parsed corpus against a frozen grounded-QA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.schemas import EvaluationDataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    corpus_bytes, dataset_bytes = args.corpus.read_bytes(), args.dataset.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    dataset = EvaluationDataset.from_json(dataset_bytes.decode())
    corpus.validate_dataset(dataset)
    print(
        json.dumps(
            {
                "valid": True,
                "corpus_version": corpus.version,
                "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.version,
                "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "examples": len(dataset.examples),
                "documents": len(corpus.documents),
                "chunks": len(corpus.chunks),
            },
            sort_keys=True,
        )
    )
    return 0
