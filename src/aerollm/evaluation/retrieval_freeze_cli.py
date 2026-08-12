"""Freeze an approved retrieval-test review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.retrieval_freeze import freeze_review_packet, write_frozen_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/corpora/ntsb-pilot-v1.json")
    )
    parser.add_argument(
        "--review", type=Path,
        default=Path("data/evaluation/retrieval_test_v1.review.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/retrieval_test_v1.json")
    )
    parser.add_argument(
        "--digest-output", type=Path,
        default=Path("data/evaluation/retrieval_test_v1.sha256"),
    )
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args(argv)
    try:
        corpus = CorpusManifest.from_json(args.corpus.read_text(encoding="utf-8"))
        dataset = freeze_review_packet(corpus, args.review, version=args.version)
        digest = write_frozen_dataset(dataset, args.output, args.digest_output)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "examples": len(dataset.examples),
                "output": args.output.as_posix(),
                "sha256": digest,
                "split": "test",
                "version": dataset.version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
