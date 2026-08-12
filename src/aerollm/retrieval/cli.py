"""Run the deterministic lexical retrieval baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.schemas import EvaluationDataset
from aerollm.retrieval.bm25 import BM25Index
from aerollm.retrieval.evaluation import evaluate_retrieval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    dataset = EvaluationDataset.from_json(args.dataset.read_text(encoding="utf-8"))
    corpus.validate_dataset(dataset)
    index = BM25Index(corpus.chunks, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest())
    report = evaluate_retrieval(dataset, index, corpus.chunks, k=args.k)
    if args.json_output:
        args.json_output.write_text(report.to_json(), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(report.to_markdown(), encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(report.to_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
