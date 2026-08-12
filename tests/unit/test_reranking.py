from collections.abc import Sequence

from aerollm.common.documents import Chunk
from aerollm.retrieval.rerank import RerankedIndex
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult


class Candidates:
    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        hits = tuple(
            RetrievalHit(chunk_id, rank, float(3 - rank))
            for rank, chunk_id in enumerate(("low", "high")[:k], start=1)
        )
        return RetrievalResult(query, "candidate-index", hits, 0.0)


class Scorer:
    model_id = "fixture/reranker"
    model_revision = "revision"

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        return tuple(10.0 if passage == "relevant passage" else -2.0 for passage in passages)


def test_cross_encoder_reorders_fixed_candidates() -> None:
    chunks = (
        _chunk("low", "irrelevant passage", 0),
        _chunk("high", "relevant passage", 20),
    )
    reranker = RerankedIndex(
        Candidates(), chunks, Scorer(), candidate_index_id="candidate-index", candidate_k=2,
    )

    result = reranker.search("question", k=2)

    assert [hit.chunk_id for hit in result.hits] == ["high", "low"]
    assert result.hits[0].score > result.hits[1].score


def _chunk(chunk_id: str, text: str, start: int) -> Chunk:
    base = Chunk.create(
        document_id="doc", text=text, start=start, end=start + len(text),
        page_start=1, page_end=1,
    )
    object.__setattr__(base, "chunk_id", chunk_id)
    return base
