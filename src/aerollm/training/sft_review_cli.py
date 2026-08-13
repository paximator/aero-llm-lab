"""Display one compact SFT record for qualitative review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--record", required=True)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    matches = [item for item in dataset["records"] if item["record_id"] == args.record]
    if not matches:
        raise ValueError(f"unknown SFT record: {args.record}")
    record = matches[0]
    messages = {message["role"]: message["content"] for message in record["messages"]}
    assistant = json.loads(messages["assistant"])
    print(f"Record: {record['record_id']}")
    print(f"Event family: {record['event_family_id']}")
    print(f"Chunk(s): {', '.join(record['source_chunk_ids'])}")
    print(f"Dataset review: {dataset['review']['comment']}")
    print("\nInstruction:\n" + messages["user"])
    print("\nExpected answer:\n" + str(assistant["answer"]))
    print("\nExpected citations:")
    for citation in assistant["citations"]:
        print(f"- {citation['chunk_id']}: {citation['quote']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
