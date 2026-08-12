from aerollm.retrieval.hybrid import HybridIndex
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult


class FixedRetriever:
    def __init__(self, index_id: str, chunks: tuple[str, ...]) -> None:
        self.index_id, self.chunks = index_id, chunks

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        return RetrievalResult(
            query, self.index_id,
            tuple(RetrievalHit(chunk, rank, float(1 / rank)) for rank, chunk in enumerate(
                self.chunks[:k], start=1
            )),
            0.0,
        )


def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    lexical = FixedRetriever("lexical", ("shared", "lexical-only"))
    dense = FixedRetriever("dense", ("dense-only", "shared"))
    hybrid = HybridIndex(
        lexical, dense, lexical_index_id="lexical", dense_index_id="dense",
        rrf_constant=1, candidate_k=2,
    )

    result = hybrid.search("question", k=3)

    assert result.hits[0].chunk_id == "shared"
    assert result.index_id == hybrid.manifest.index_id
