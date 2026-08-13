"""Validate and score evaluation-suite-v2 development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerollm.evaluation.task_scoring import evaluate_task_answers, validate_task_annotation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    examples = dataset["examples"]
    for example in examples:
        validate_task_annotation(example)
    if args.predictions is None:
        print(f"validated {len(examples)} task-specific annotations")
        return 0
    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = {item["example_id"]: item.get("answer") for item in raw_predictions}
    report = (
        json.dumps(evaluate_task_answers(examples, predictions), indent=2, sort_keys=True) + "\n"
    )
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
