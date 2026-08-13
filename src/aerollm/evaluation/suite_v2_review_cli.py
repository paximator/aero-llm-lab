"""Generate the evaluation-suite-v2 human-review workbook."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.suite_v2_review import (
    build_suite_v2_review,
    write_suite_v2_public_bundle,
    write_suite_v2_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpora/ntsb-corpus-v2.json"),
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path("configs/evaluation/evaluation_suite_v2.proposal.toml"),
    )
    parser.add_argument(
        "--development-draft",
        type=Path,
        default=Path("configs/evaluation/evaluation_suite_v2.dev_draft.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/private_evaluation/evaluation_suite_v2_test_gold.json"),
    )
    parser.add_argument(
        "--public-directory",
        type=Path,
        default=Path("data/evaluation/v2"),
    )
    parser.add_argument(
        "--review-workbook",
        type=Path,
        help="Publish an existing reviewed workbook instead of rebuilding it.",
    )
    parser.add_argument(
        "--frozen-selection",
        type=Path,
        default=Path("artifacts/pilot/corpus_v2_selection.frozen.json"),
    )
    args = parser.parse_args()
    if args.review_workbook is None:
        corpus = CorpusManifest.from_json(args.corpus.read_text(encoding="utf-8"))
        packet = build_suite_v2_review(
            corpus,
            args.proposal,
            args.development_draft,
            args.frozen_selection,
        )
        write_suite_v2_review(packet, args.output)
    else:
        decoded = json.loads(args.review_workbook.read_text(encoding="utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("review workbook must be a JSON object")
        packet = decoded
    write_suite_v2_public_bundle(packet, args.public_directory)
    private_location = args.output if args.review_workbook is None else args.review_workbook
    print(
        f"wrote public suite-v2 metadata to {args.public_directory}; "
        f"private workbook: {private_location}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
