"""Build and freeze the retrieval development dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from aerollm.data.snapshots import atomic_write
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.retrieval_dataset import (
    build_development_dataset,
    build_test_review_packet,
    write_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("artifacts/corpora/ntsb-pilot-v1.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/evaluation/retrieval_dev_v1.toml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/evaluation/retrieval_dev_v1.json"),
    )
    parser.add_argument(
        "--test-draft-config", type=Path,
        default=Path("configs/evaluation/retrieval_test_v1.draft.toml"),
    )
    parser.add_argument(
        "--test-review-output", type=Path,
        default=Path("data/evaluation/retrieval_test_v1.review.json"),
    )
    arguments = parser.parse_args(argv)
    try:
        corpus = CorpusManifest.from_json(arguments.corpus.read_text(encoding="utf-8"))
        dataset = build_development_dataset(corpus, arguments.config)
        digest = write_dataset(dataset, arguments.output)
        review = build_test_review_packet(corpus, arguments.test_draft_config)
        review_content = (json.dumps(review, indent=2, sort_keys=True) + "\n").encode()
        atomic_write(arguments.test_review_output, review_content)
        review_digest = hashlib.sha256(review_content).hexdigest()
    except (
        KeyError, OSError, TypeError, ValueError,
        json.JSONDecodeError, tomllib.TOMLDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": arguments.output.as_posix(), "sha256": digest,
                "examples": len(dataset.examples),
                "test_review_output": arguments.test_review_output.as_posix(),
                "test_review_sha256": review_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
