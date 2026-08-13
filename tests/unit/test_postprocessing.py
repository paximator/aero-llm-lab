import json

import pytest

from aerollm.postprocessing import (
    Citation,
    PostprocessorV1,
    RawGeneration,
    RetrievedEvidence,
    normalize_whitespace,
    postprocess,
)


def _evidence() -> tuple[RetrievedEvidence, ...]:
    return (RetrievedEvidence("chunk-1", "The aircraft landed safely."),)


def _payload(**updates: object) -> str:
    value = {
        "answer": "The aircraft landed safely.",
        "citations": [{"chunk_id": "chunk-1", "quote": "landed safely"}],
        "abstained": False,
        "abstention_reason": None,
    }
    value.update(updates)
    return json.dumps(value)


def test_valid_json_is_normalized_and_grounded() -> None:
    raw = _payload(answer="The\u00a0 aircraft\tlanded safely.")

    result = postprocess(RawGeneration(raw), _evidence())

    assert result.answer == "The aircraft landed safely."
    assert result.citations == (Citation("chunk-1", "landed safely"),)
    assert not result.abstained
    assert result.raw_output == raw
    assert result.warnings == ()


def test_fenced_json_is_accepted() -> None:
    result = postprocess(f"```json\n{_payload()}\n```", _evidence())

    assert not result.abstained


@pytest.mark.parametrize("raw, warning", [("", "empty_output"), ("plain answer", "invalid_json")])
def test_empty_and_plain_text_fail_closed(raw: str, warning: str) -> None:
    result = postprocess(raw, _evidence())

    assert result.abstained
    assert result.abstention_reason == "insufficient evidence"
    assert result.raw_output == raw
    assert warning in result.warnings


def test_overlong_output_is_not_parsed() -> None:
    result = PostprocessorV1(max_output_chars=4).process("12345", _evidence())

    assert result.abstained
    assert result.warnings == ("output_too_long",)


def test_unknown_or_non_verbatim_citations_force_abstention() -> None:
    raw = _payload(citations=[{"chunk_id": "missing", "quote": "landed safely"}])

    result = postprocess(raw, _evidence())

    assert result.abstained
    assert result.citations == ()
    assert result.warnings == ("invalid_citation", "insufficient_evidence")


def test_valid_citations_survive_invalid_ones_with_warning() -> None:
    raw = _payload(
        citations=[
            {"chunk_id": "missing", "quote": "not evidence"},
            {"chunk_id": "chunk-1", "quote": "landed safely"},
        ]
    )

    result = postprocess(raw, _evidence())

    assert not result.abstained
    assert result.warnings == ("invalid_citation",)


def test_explicit_abstention_is_normalized_and_preserved() -> None:
    raw = _payload(answer=None, citations=[], abstained=True, abstention_reason="Not\u00a0enough")

    result = postprocess(raw, ())

    assert result.abstained
    assert result.abstention_reason == "Not enough"
    assert result.warnings == ()


def test_malformed_contract_abstains() -> None:
    result = postprocess('{"answer": "unsupported"}', _evidence())

    assert result.abstained
    assert result.warnings == ("malformed_output",)


def test_unicode_whitespace_normalization_preserves_line_boundaries() -> None:
    assert normalize_whitespace(" A\u00a0 B\r\n\tC ") == "A B\nC"


def test_duplicate_evidence_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        postprocess(_payload(), (*_evidence(), *_evidence()))
