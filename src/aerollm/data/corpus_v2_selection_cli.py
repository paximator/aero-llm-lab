"""Generate a metadata-only corpus-v2 event selection review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aerollm.data.corpus_v2_selection import (
    SelectionConfig,
    build_review,
    cached_candidates,
    discover_candidates,
    select_candidates,
    write_review,
)
from aerollm.evaluation.corpus import CorpusManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/corpus_v2_selection.toml"),
    )
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--ntsb-config", type=Path, default=Path("configs/data/ntsb.toml"))
    parser.add_argument(
        "--discovery-failures",
        type=Path,
        default=Path("artifacts/pilot/corpus_v2_discovery.failures.json"),
    )
    args = parser.parse_args(argv)

    config = SelectionConfig.load(args.config)
    corpus_bytes = args.source_corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode("utf-8"))
    excluded = {source.event_family_id for source in corpus.sources}
    if args.discover:
        failures = discover_candidates(config, args.ntsb_config)
        args.discovery_failures.parent.mkdir(parents=True, exist_ok=True)
        args.discovery_failures.write_text(
            json.dumps({"failures": failures}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assignments = select_candidates(
        cached_candidates(config.snapshot_root), excluded_families=excluded,
        config=config,
    )
    review = build_review(
        assignments, config=config, excluded_families=excluded,
        source_corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
    )
    output = args.output or config.review_output
    write_review(output, review)
    print(f"wrote {len(assignments)} metadata-only candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
