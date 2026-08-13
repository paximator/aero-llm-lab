from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerollm.common.documents import Chunk
from aerollm.retrieval.schemas import RetrievalHit, RetrievalResult
from aerollm.serving.bootstrap import ProductionConfig, build_production_retriever


class FakeDenseIndex:
    def __init__(self, corpus_sha256: str, chunk_ids: tuple[str, ...]) -> None:
        self.manifest = SimpleNamespace(
            corpus_sha256=corpus_sha256, chunk_ids=chunk_ids, index_id="dense-fixture",
        )

    def search(self, query: str, *, k: int = 10) -> RetrievalResult:
        return RetrievalResult(
            query, self.manifest.index_id,
            (RetrievalHit("relevant", 1, 1.0), RetrievalHit("other", 2, 0.5))[:k], 0.0,
        )


class FakeScorer:
    model_id = "fixture/reranker"
    model_revision = "revision"

    def score(self, query: str, passages: list[str]) -> tuple[float, ...]:
        del query
        return tuple(10.0 if passage == "relevant evidence" else -1.0 for passage in passages)


def test_bm25_mode_is_an_explicit_model_free_fallback() -> None:
    config = replace(_config(), retrieval_mode="bm25")
    corpus = SimpleNamespace(chunks=(_chunk("relevant", "relevant evidence", 0),))

    retriever = build_production_retriever(config, corpus, "a" * 64)
    passages = retriever.retrieve("relevant", top_k=1)

    assert passages[0].chunk_id == "relevant"
    assert passages[0].text == "relevant evidence"


def test_hybrid_reranked_mode_composes_existing_indexes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from aerollm.retrieval.dense import DenseIndex
    from aerollm.retrieval.dense_transformers import TransformersDenseEncoder
    from aerollm.retrieval.rerank_transformers import TransformersCrossEncoder

    corpus_sha256 = "b" * 64
    chunks = (
        _chunk("other", "other evidence", 0),
        _chunk("relevant", "relevant evidence", 20),
    )
    corpus = SimpleNamespace(chunks=chunks)
    chunk_ids = tuple(sorted(chunk.chunk_id for chunk in chunks))
    monkeypatch.setattr(
        TransformersDenseEncoder, "from_local_path", classmethod(lambda cls, *a, **kw: object()),
    )
    monkeypatch.setattr(
        DenseIndex, "load", classmethod(
            lambda cls, path, encoder: FakeDenseIndex(corpus_sha256, chunk_ids)
        ),
    )
    monkeypatch.setattr(
        TransformersCrossEncoder, "from_local_path",
        classmethod(lambda cls, *a, **kw: FakeScorer()),
    )

    retriever = build_production_retriever(_config(), corpus, corpus_sha256)
    passages = retriever.retrieve("question", top_k=2)

    assert [passage.chunk_id for passage in passages] == ["relevant", "other"]


def test_hybrid_mode_rejects_dense_index_for_another_corpus(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from aerollm.retrieval.dense import DenseIndex
    from aerollm.retrieval.dense_transformers import TransformersDenseEncoder

    chunk = _chunk("relevant", "relevant evidence", 0)
    monkeypatch.setattr(
        TransformersDenseEncoder, "from_local_path", classmethod(lambda cls, *a, **kw: object()),
    )
    monkeypatch.setattr(
        DenseIndex, "load", classmethod(
            lambda cls, path, encoder: FakeDenseIndex("c" * 64, (chunk.chunk_id,))
        ),
    )

    with pytest.raises(ValueError, match="corpus digest"):
        build_production_retriever(_config(), SimpleNamespace(chunks=(chunk,)), "d" * 64)


def _config() -> ProductionConfig:
    return ProductionConfig.from_toml(Path("configs/serving/production_v1.toml"))


def _chunk(chunk_id: str, text: str, start: int) -> Chunk:
    chunk = Chunk.create(
        document_id="document", text=text, start=start, end=start + len(text),
        page_start=1, page_end=1,
    )
    object.__setattr__(chunk, "chunk_id", chunk_id)
    return chunk
