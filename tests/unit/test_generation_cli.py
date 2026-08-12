import pytest

from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.evaluation.generation_cli import _require_allowed_split
from aerollm.evaluation.schemas import EvaluationDataset, ExampleProvenance, GroundedQAExample


def _dataset(split: Split) -> EvaluationDataset:
    reviewers = ("reviewer",) if split is Split.TEST else ()
    example = GroundedQAExample(
        "example", "Question?", "report", "event", "family", split, True, "Answer.", (),
        (EvidenceSpan("chunk", "quote"),),
        ExampleProvenance("author", reviewers, ("report",), False),
    )
    return EvaluationDataset("dataset", "1", (example,))


def test_generation_development_gate_accepts_development() -> None:
    assert _require_allowed_split(
        _dataset(Split.DEVELOPMENT), {"allowed_split": "development"},
        dataset_bytes=b"development", allow_frozen_test=False,
    ) is Split.DEVELOPMENT


def test_generation_development_gate_rejects_frozen_test() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _require_allowed_split(
            _dataset(Split.TEST), {"allowed_split": "development"},
            dataset_bytes=b"test", allow_frozen_test=False,
        )


def test_generation_test_gate_requires_explicit_flag_and_digest() -> None:
    import hashlib

    config = {
        "allowed_split": "test",
        "locked_test_dataset_sha256": hashlib.sha256(b"test").hexdigest(),
    }
    with pytest.raises(ValueError, match="--allow-frozen-test"):
        _require_allowed_split(
            _dataset(Split.TEST), config, dataset_bytes=b"test", allow_frozen_test=False,
        )
    assert _require_allowed_split(
        _dataset(Split.TEST), config, dataset_bytes=b"test", allow_frozen_test=True,
    ) is Split.TEST
