"""Command-line entry point for frozen corpus construction."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from aerollm.data.corpus import CorpusConfig, build_corpus, write_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic report-level corpus")
    parser.add_argument("source_manifests", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data/corpus.toml"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = CorpusConfig.load(arguments.config)
        result = build_corpus(arguments.source_manifests, config)
        output = arguments.output or config.output
        digest, build_path = write_corpus(result, output)
    except (
        KeyError, OSError, TypeError, ValueError,
        json.JSONDecodeError, tomllib.TOMLDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = {
        "output": output.as_posix(), "build_manifest": build_path.as_posix(),
        "sha256": digest, "documents": len(result.corpus.documents),
        "chunks": len(result.corpus.chunks),
        "warnings": len(result.validation.warnings),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
