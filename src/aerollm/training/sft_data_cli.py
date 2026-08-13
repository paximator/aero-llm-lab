"""Build the 50-record train-only evidence-linked SFT validation dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.training.sft_data import build_sft_dataset, write_sft_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/training/sft_v1_50.json"))
    parser.add_argument("--records", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode("utf-8"))
    dataset = build_sft_dataset(
        corpus,
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        target_records=args.records,
        max_tokens=args.max_tokens,
    )
    write_sft_dataset(dataset, args.output)
    print(f"wrote {len(dataset['records'])} train-only SFT records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
