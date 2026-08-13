import json
from collections import Counter
from pathlib import Path


def test_repository_corrective_dataset_is_balanced_and_reviewed() -> None:
    dataset = json.loads(Path("data/training/sft_v2_90.json").read_text(encoding="utf-8"))
    corrective = dataset["records"][50:]

    assert len(dataset["records"]) == 90
    assert dataset["status"] == "model_assisted_review_passed"
    assert dataset["review"]["reviewed_records"] == 40
    assert Counter(record["task_type"] for record in corrective) == {
        "abstention": 10,
        "causal": 10,
        "multi_evidence": 10,
        "exact_citation": 10,
    }
    assert all(len(record["source_chunk_ids"]) == 2 for record in corrective[10:20])
