"""Build or evaluate a persistent exact dense-retrieval index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.schemas import EvaluationDataset
from aerollm.retrieval.dense import DenseIndex
from aerollm.retrieval.dense_transformers import TransformersDenseEncoder
from aerollm.retrieval.evaluation import evaluate_retrieval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    encoder = TransformersDenseEncoder.from_local_path(
        args.model, model_id=args.model_id, model_revision=args.model_revision,
        device=args.device, batch_size=args.batch_size,
    )
    if args.build:
        index = DenseIndex.build(
            corpus.chunks, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
            encoder=encoder,
        )
        index.save(args.index)
    else:
        index = DenseIndex.load(args.index, encoder)
    if args.dataset:
        dataset = EvaluationDataset.from_json(args.dataset.read_text(encoding="utf-8"))
        corpus.validate_dataset(dataset)
        report = evaluate_retrieval(dataset, index, corpus.chunks, k=args.k)  # type: ignore[arg-type]
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(report.to_json(), encoding="utf-8")
        if args.markdown_output:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(report.to_markdown(), encoding="utf-8")
        if not args.json_output and not args.markdown_output:
            print(report.to_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
