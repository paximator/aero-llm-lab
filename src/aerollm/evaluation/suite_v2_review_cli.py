"""Generate the evaluation-suite-v2 human-review workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.suite_v2_review import build_suite_v2_review, write_suite_v2_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("artifacts/corpora/ntsb-corpus-v2.json"),
    )
    parser.add_argument(
        "--proposal", type=Path,
        default=Path("configs/evaluation/evaluation_suite_v2.proposal.toml"),
    )
    parser.add_argument(
        "--development-draft", type=Path,
        default=Path("configs/evaluation/evaluation_suite_v2.dev_draft.toml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/evaluation/evaluation_suite_v2.review.json"),
    )
    parser.add_argument(
        "--frozen-selection", type=Path,
        default=Path("artifacts/pilot/corpus_v2_selection.frozen.json"),
    )
    args = parser.parse_args()
    corpus = CorpusManifest.from_json(args.corpus.read_text(encoding="utf-8"))
    packet = build_suite_v2_review(
        corpus, args.proposal, args.development_draft, args.frozen_selection,
    )
    write_suite_v2_review(packet, args.output)
    print(f"wrote {len(packet['examples'])} review slots to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
