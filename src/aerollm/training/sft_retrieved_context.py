"""Build a tiny train-only retrieved-context SFT gate from reviewed SFT v1 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from aerollm.evaluation.sft_comparison import _SYSTEM_CONCISE_V2

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{4,}")
_STOP = {
    "about",
    "after",
    "aircraft",
    "airplane",
    "aviation",
    "before",
    "during",
    "flight",
    "report",
    "stated",
    "their",
    "there",
    "these",
    "which",
}


def build(
    parent: Mapping[str, object], *, answerable: int = 8, abstentions: int = 4
) -> dict[str, object]:
    """Create a deterministic micro-dataset without introducing unreviewed facts."""
    records = parent.get("records")
    if not isinstance(records, list) or answerable < 1 or abstentions < 1:
        raise ValueError("parent records and positive gate sizes are required")
    eligible = [record for record in records if _eligible(record)]
    if len(eligible) < answerable + 2:
        raise ValueError("not enough concise reviewed parent records")
    selected = eligible[:answerable]
    built = []
    for index, record in enumerate(selected):
        distractors = [eligible[(index + offset) % len(eligible)] for offset in (1, 2)]
        built.append(_answerable(record, distractors))
    for index in range(abstentions):
        contexts = [eligible[(answerable + index + offset) % len(eligible)] for offset in range(3)]
        built.append(_abstention(index, contexts))
    result = {
        "schema_version": 1,
        "dataset_id": "ntsb-retrieved-context-sft-micro-v1",
        "version": "1.0.0",
        "status": "automated_gate_only",
        "parent_dataset_sha256": _digest(parent),
        "split": "train",
        "selection": {
            "method": "reviewed_parent_with_two_train_distractors_v1",
            "answerable": answerable,
            "abstentions": abstentions,
        },
        "records": built,
    }
    validate(result)
    return result


def validate(dataset: Mapping[str, object]) -> None:
    """Validate schema, exact citations, distractors, abstentions, and parent lineage."""
    if dataset.get("split") != "train":
        raise ValueError("retrieved-context SFT gate must be train-only")
    records = dataset.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("retrieved-context records are required")
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "record_id",
            "event_family_id",
            "source_chunk_ids",
            "messages",
        }:
            raise ValueError("invalid retrieved-context record")
        record_id = str(record["record_id"])
        if record_id in ids:
            raise ValueError("duplicate retrieved-context record")
        ids.add(record_id)
        chunk_ids = record["source_chunk_ids"]
        messages = record["messages"]
        if not isinstance(chunk_ids, list) or len(chunk_ids) != 3:
            raise ValueError("each record requires one context and two distractors")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError("each record requires a complete chat")
        user = str(messages[1]["content"])
        payload = json.loads(str(messages[2]["content"]))
        if payload["abstained"]:
            if payload != {
                "answer": None,
                "citations": [],
                "abstained": True,
                "abstention_reason": "The supplied chunks do not contain that information.",
            }:
                raise ValueError("invalid retrieved-context abstention")
        else:
            if len(payload["answer"].split()) > 40 or len(payload["citations"]) != 1:
                raise ValueError("answer must be concise and singly cited")
            citation = payload["citations"][0]
            if citation["chunk_id"] not in chunk_ids or citation["quote"] not in user:
                raise ValueError("citation must resolve exactly to supplied context")


def _eligible(record: object) -> bool:
    if not isinstance(record, Mapping):
        return False
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return False
    payload = json.loads(str(messages[2]["content"]))
    return len(str(payload["answer"]).split()) <= 35


def _answerable(
    target: Mapping[str, object], distractors: list[Mapping[str, object]]
) -> dict[str, object]:
    target_payload = json.loads(str(target["messages"][2]["content"]))
    answer = str(target_payload["answer"])
    terms = [word for word in _WORD.findall(answer) if word.casefold() not in _STOP]
    clue = ", ".join(list(dict.fromkeys(word.casefold() for word in terms))[:3])
    contexts = [target, *distractors]
    user = f"What does the report state about {clue}?\n\n{_contexts(contexts)}"
    assistant = json.dumps(target_payload, ensure_ascii=False, separators=(",", ":"))
    return _record(target, contexts, user, assistant, "answerable")


def _abstention(index: int, contexts: list[Mapping[str, object]]) -> dict[str, object]:
    user = f"What was the flight attendant's favorite color?\n\n{_contexts(contexts)}"
    assistant = json.dumps(
        {
            "answer": None,
            "citations": [],
            "abstained": True,
            "abstention_reason": "The supplied chunks do not contain that information.",
        },
        separators=(",", ":"),
    )
    return _record(contexts[0], contexts, user, assistant, f"abstain-{index}")


def _contexts(records: list[Mapping[str, object]]) -> str:
    rendered = []
    for record in records:
        chunk_id = str(record["source_chunk_ids"][0])
        content = str(record["messages"][1]["content"])
        excerpt = content.split("\n<chunk", 1)[1].split("\n", 1)[1].rsplit("\n</chunk>", 1)[0]
        rendered.append(f'<chunk id="{chunk_id}">\n{excerpt}\n</chunk>')
    return "\n\n".join(rendered)


def _record(
    target: Mapping[str, object],
    contexts: list[Mapping[str, object]],
    user: str,
    assistant: str,
    suffix: str,
) -> dict[str, object]:
    parent_id = str(target["record_id"])
    identity = hashlib.sha256(f"{parent_id}\0{suffix}".encode()).hexdigest()[:16]
    return {
        "record_id": f"sft-rag-{identity}",
        "event_family_id": target["event_family_id"],
        "source_chunk_ids": [str(record["source_chunk_ids"][0]) for record in contexts],
        "messages": [
            {"role": "system", "content": _SYSTEM_CONCISE_V2},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answerable", type=int, default=8)
    parser.add_argument("--abstentions", type=int, default=4)
    args = parser.parse_args(argv)
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    dataset = build(parent, answerable=args.answerable, abstentions=args.abstentions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(dataset['records'])} train-only records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
