import copy
import json
from pathlib import Path

import pytest

from aerollm.training.sft_retrieved_context import build, validate


def _parent() -> dict[str, object]:
    root = Path(__file__).parents[2]
    return json.loads((root / "data/training/sft_v1_50.json").read_text(encoding="utf-8"))


def test_builder_creates_train_only_answerable_and_abstention_gate() -> None:
    dataset = build(_parent(), answerable=3, abstentions=2)

    assert dataset["split"] == "train"
    assert len(dataset["records"]) == 5
    assert all(len(record["source_chunk_ids"]) == 3 for record in dataset["records"])
    payloads = [json.loads(record["messages"][2]["content"]) for record in dataset["records"]]
    assert sum(payload["abstained"] for payload in payloads) == 2
    validate(dataset)


def test_validator_rejects_citation_outside_supplied_context() -> None:
    dataset = build(_parent(), answerable=2, abstentions=1)
    broken = copy.deepcopy(dataset)
    payload = json.loads(broken["records"][0]["messages"][2]["content"])
    payload["citations"][0]["quote"] = "not in any supplied chunk"
    broken["records"][0]["messages"][2]["content"] = json.dumps(payload)

    with pytest.raises(ValueError, match="citation"):
        validate(broken)
