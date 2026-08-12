import hashlib
from collections.abc import Sequence
from pathlib import Path

from aerollm.common.documents import Chunk
from aerollm.retrieval.dense import DenseIndex


class FixtureEncoder:
    model_id = "fixture/encoder"
    model_revision = "abc123"
    dimension = 2
    query_prefix = "query: "
    passage_prefix = "passage: "

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def encode_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    @staticmethod
    def _vector(text: str) -> tuple[float, float]:
        return (1.0, 0.0) if "aircraft" in text else (0.0, 1.0)


def test_dense_index_build_search_save_and_load(tmp_path: Path) -> None:
    chunks = (
        _chunk("aircraft collision", 0, 1),
        _chunk("weather conditions", 20, 2),
    )
    encoder = FixtureEncoder()
    index = DenseIndex.build(chunks, corpus_sha256="a" * 64, encoder=encoder)

    result = index.search("which aircraft?", k=1)
    assert result.hits[0].chunk_id == chunks[0].chunk_id

    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, encoder)
    assert loaded.manifest == index.manifest
    assert loaded.search("which aircraft?", k=1).hits[0].chunk_id == chunks[0].chunk_id
    assert hashlib.sha256((tmp_path / "vectors.f32").read_bytes()).hexdigest()


def test_dense_index_identity_is_stable_under_chunk_order() -> None:
    chunks = (
        _chunk("aircraft collision", 0, 1),
        _chunk("weather conditions", 20, 2),
    )
    encoder = FixtureEncoder()

    first = DenseIndex.build(chunks, corpus_sha256="b" * 64, encoder=encoder)
    second = DenseIndex.build(tuple(reversed(chunks)), corpus_sha256="b" * 64, encoder=encoder)

    assert first.manifest == second.manifest


def _chunk(text: str, start: int, page: int) -> Chunk:
    return Chunk.create(
        document_id="doc", text=text, start=start, end=start + len(text),
        page_start=page, page_end=page,
    )
