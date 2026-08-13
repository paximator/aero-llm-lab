"""Build the 90-record corrective SFT development dataset."""

import argparse
import hashlib
import json
from pathlib import Path

from aerollm.evaluation.corpus import CorpusManifest
from aerollm.training.sft_corrective import build_corrective_dataset, write_corrective_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--base", type=Path, default=Path("data/training/sft_v1_50.json"))
    parser.add_argument("--output", type=Path, default=Path("data/training/sft_v2_90.json"))
    args = parser.parse_args(argv)
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    base = json.loads(args.base.read_text(encoding="utf-8"))
    dataset = build_corrective_dataset(
        base, corpus, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest()
    )
    write_corrective_dataset(dataset, args.output)
    print(f"wrote {len(dataset['records'])} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
