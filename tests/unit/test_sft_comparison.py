from aerollm.evaluation.sft_comparison import (
    aggregate,
    build_messages,
    classify_failure,
    failure_counts,
)


def test_comparison_prompt_uses_reviewed_evidence() -> None:
    messages, evidence = build_messages(
        {
            "question": "What happened?",
            "evidence": [{"chunk_id": "chunk-1", "quote": "The airplane landed."}],
        }
    )

    assert messages[1]["content"].endswith('<chunk id="chunk-1">\nThe airplane landed.\n</chunk>')
    assert evidence[0].chunk_id == "chunk-1"


def test_comparison_aggregate_keeps_grounding_separate() -> None:
    summary = aggregate(
        [
            {
                "correct": True,
                "valid_grounded_output": False,
                "abstained": True,
                "latency_ms": 10,
                "warnings": [],
            },
            {
                "correct": False,
                "valid_grounded_output": True,
                "abstained": False,
                "latency_ms": 20,
                "warnings": [],
            },
        ]
    )

    assert summary["task_accuracy_after_fail_closed"] == 0.5
    assert summary["valid_grounded_output_rate"] == 0.5
    assert summary["format_failure_rate"] == 0.0
    assert summary["mean_latency_ms"] == 15


def test_comparison_merges_quotes_from_the_same_chunk() -> None:
    _, evidence = build_messages(
        {
            "question": "What happened?",
            "evidence": [
                {"chunk_id": "chunk-1", "quote": "First fact."},
                {"chunk_id": "chunk-1", "quote": "Second fact."},
            ],
        }
    )

    assert len(evidence) == 1
    assert evidence[0].text == "First fact.\nSecond fact."


def test_comparison_classifies_truncated_json_separately_from_plain_text() -> None:
    truncated = {
        "raw_output": '{"answer":"unfinished',
        "warnings": ["invalid_json"],
        "completion_tokens": 192,
    }
    plain = {
        "raw_output": "A plain answer.",
        "warnings": ["invalid_json"],
        "completion_tokens": 12,
    }

    assert classify_failure(truncated, max_new_tokens=192) == "generation_truncation"
    assert classify_failure(plain, max_new_tokens=192) == "non_json_output"
    assert failure_counts([truncated, plain], max_new_tokens=192) == {
        "generation_truncation": 1,
        "non_json_output": 1,
    }
