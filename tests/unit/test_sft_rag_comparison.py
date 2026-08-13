from aerollm.common.documents import Chunk
from aerollm.evaluation.sft_rag_comparison import (
    prepare_retrieved_example,
    select_context_passage,
)
from aerollm.retrieval.schemas import RetrievalHit


def test_prepared_example_uses_retrieved_text_and_reports_gold_hit() -> None:
    chunk = Chunk.create(
        document_id="doc-1",
        text="Retrieved report evidence.",
        start=0,
        end=26,
        page_start=1,
        page_end=1,
    )
    example = {
        "example_id": "example-1",
        "question": "What happened?",
        "evidence": [{"chunk_id": chunk.chunk_id, "quote": "report evidence"}],
    }

    prepared, diagnostic = prepare_retrieved_example(
        example,
        (RetrievalHit(chunk.chunk_id, 1, 0.9),),
        {chunk.chunk_id: chunk},
        context_mode="passage",
    )

    assert prepared["evidence"] == [
        {"chunk_id": chunk.chunk_id, "quote": "Retrieved report evidence."}
    ]
    assert diagnostic == {
        "example_id": "example-1",
        "retrieved_chunk_ids": [chunk.chunk_id],
        "gold_hit_at_3": True,
    }
    assert example["evidence"][0]["quote"] == "report evidence"


def test_context_passage_prefers_query_terms_and_preserves_exact_text() -> None:
    text = (
        "The airplane departed normally. "
        "Seven passengers and both pilots died in the accident. "
        "The weather was cloudy."
    )

    passage = select_context_passage("How many passengers died?", text)

    assert passage == "Seven passengers and both pilots died in the accident."
    assert passage in text
