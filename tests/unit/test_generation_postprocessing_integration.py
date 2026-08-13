import json

import pytest

from aerollm.common.documents import Chunk
from aerollm.generation import (
    FakeBackend,
    GenerationRequest,
    GroundedPredictionV1,
    build_grounded_prediction,
    select_retrieved_chunks,
)
from aerollm.retrieval import RetrievalHit, RetrievalResult


def _chunk(text: str, start: int = 0) -> Chunk:
    return Chunk.create(
        document_id="doc-1",
        text=text,
        start=start,
        end=start + len(text),
        page_start=1,
        page_end=1,
    )


def _retrieval(*chunks: Chunk) -> RetrievalResult:
    return RetrievalResult(
        query="What happened?",
        index_id="index-v1",
        hits=tuple(
            RetrievalHit(chunk.chunk_id, rank, float(1 / rank))
            for rank, chunk in enumerate(chunks, start=1)
        ),
        latency_ms=2.0,
    )


def _generate(text: str, context: tuple[str, ...]):
    backend = FakeBackend(responder=lambda request: text)
    return backend.generate(
        GenerationRequest(
            prompt="What happened?",
            context=context,
            request_id="qa-1",
            prompt_version="grounded-qa-v1",
        )
    )


def test_generation_is_postprocessed_into_unified_prediction() -> None:
    chunk = _chunk("The aircraft landed safely.")
    raw = json.dumps(
        {
            "answer": "The aircraft landed safely.",
            "citations": [{"chunk_id": chunk.chunk_id, "quote": "landed safely"}],
            "abstained": False,
            "abstention_reason": None,
        }
    )
    generation = _generate(raw, (chunk.text,))

    prediction = build_grounded_prediction(generation, _retrieval(chunk), (chunk,))

    assert isinstance(prediction, GroundedPredictionV1)
    assert prediction.output.answer == "The aircraft landed safely."
    assert prediction.selected_chunk_ids == (chunk.chunk_id,)
    assert prediction.output.raw_output == raw
    serialized = prediction.to_dict()
    assert serialized["retrieval"]["hits"][0]["score"] == 1.0
    assert serialized["generation_trace"]["request_id"] == "qa-1"
    assert serialized["generation_trace"]["prompt_tokens"] > 0


def test_malformed_generation_becomes_auditable_abstention() -> None:
    chunk = _chunk("The aircraft landed safely.")
    generation = _generate("not json", (chunk.text,))

    prediction = build_grounded_prediction(generation, _retrieval(chunk), (chunk,))

    assert prediction.output.abstained
    assert prediction.output.raw_output == "not json"
    assert prediction.output.warnings == ("invalid_json",)


def test_only_selected_chunks_can_support_citations() -> None:
    selected = _chunk("The aircraft landed safely.")
    unselected = _chunk("The pilot reported clear weather.", start=100)
    raw = json.dumps(
        {
            "answer": "The weather was clear.",
            "citations": [{"chunk_id": unselected.chunk_id, "quote": "clear weather"}],
            "abstained": False,
            "abstention_reason": None,
        }
    )

    prediction = build_grounded_prediction(
        _generate(raw, (selected.text,)), _retrieval(selected, unselected), (selected,)
    )

    assert prediction.output.abstained
    assert "invalid_citation" in prediction.output.warnings


def test_selected_chunks_must_preserve_retrieval_order() -> None:
    first = _chunk("First evidence.")
    second = _chunk("Second evidence.", start=100)

    with pytest.raises(ValueError, match="retrieval order"):
        build_grounded_prediction(
            _generate("", (second.text, first.text)),
            _retrieval(first, second),
            (second, first),
        )


def test_generation_context_must_match_selected_chunks() -> None:
    chunk = _chunk("Evidence supplied to postprocessing.")

    with pytest.raises(ValueError, match="generation context"):
        build_grounded_prediction(
            _generate("", ("Different model context.",)),
            _retrieval(chunk),
            (chunk,),
        )


def test_retrieval_hits_resolve_to_ranked_chunks() -> None:
    first = _chunk("First evidence.")
    second = _chunk("Second evidence.", start=100)
    retrieval = _retrieval(first, second)

    resolved = select_retrieved_chunks(
        retrieval, {second.chunk_id: second, first.chunk_id: first}
    )

    assert resolved == (first, second)


def test_missing_retrieved_chunk_is_rejected() -> None:
    chunk = _chunk("Evidence.")

    with pytest.raises(ValueError, match="unavailable"):
        select_retrieved_chunks(_retrieval(chunk), {})
