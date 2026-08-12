"""Command-line entry point for deterministic grounded-QA evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerollm.evaluation.runner import evaluate, predictions_from_json
from aerollm.evaluation.schemas import EvaluationDataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    dataset_bytes = args.dataset.read_bytes()
    dataset = EvaluationDataset.from_json(dataset_bytes.decode("utf-8"))
    predictions = predictions_from_json(args.predictions.read_text(encoding="utf-8"))
    report = evaluate(dataset, predictions, dataset_bytes=dataset_bytes).to_json() + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
