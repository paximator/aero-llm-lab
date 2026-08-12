"""CLI for deterministic chunk manifests."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from aerollm.common.schemas import Document
from aerollm.data.chunking import ChunkingConfig, chunk_document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk one canonical parsed document")
    parser.add_argument("document", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data/chunking.toml"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        document = Document.from_dict(json.loads(arguments.document.read_text(encoding="utf-8")))
        config = ChunkingConfig.from_dict(
            tomllib.loads(arguments.config.read_text(encoding="utf-8"))
        )
        manifest = chunk_document(document, config)
        output = arguments.output or Path("artifacts/chunks/ntsb") / (
            f"{document.document_id}.json"
        )
        digest = manifest.write(output)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = {
        "output": output.as_posix(),
        "sha256": digest,
        "chunks": len(manifest.chunks),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
