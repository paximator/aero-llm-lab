import json

import pytest

from aerollm.common.documents import ContentKind
from aerollm.common.schemas import Document, PageSpan
from aerollm.data.chunking import ChunkingConfig, ChunkManifest, chunk_document


def _document(pages: tuple[str, ...]) -> Document:
    text = ""
    spans = []
    for number, page in enumerate(pages, 1):
        if text:
            text += "\n\n"
        start = len(text)
        text += page
        spans.append(PageSpan(number, start, len(text)))
    digest = "a" * 64
    return Document(f"doc-{digest}", digest, text, tuple(spans), "fixture")


def test_chunking_is_byte_deterministic_and_round_trips(tmp_path) -> None:
    paragraph = "The aircraft landed safely after the approach. " * 12
    document = _document((f"ANALYSIS\n\n{paragraph}\n\n{paragraph}",))
    config = ChunkingConfig(300, 50, 100, 80)

    first = chunk_document(document, config)
    second = chunk_document(document, config)
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"

    assert first.write(first_path) == second.write(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert ChunkManifest.from_dict(json.loads(first_path.read_text(encoding="utf-8"))) == first


def test_chunks_match_exact_source_offsets_and_stay_within_pages() -> None:
    document = _document(("First page sentence. " * 50, "Second page sentence. " * 50))
    manifest = chunk_document(document, ChunkingConfig(240, 40, 80, 50))

    assert {chunk.page_start for chunk in manifest.chunks} == {1, 2}
    assert all(chunk.page_start == chunk.page_end for chunk in manifest.chunks)
    assert all(document.text[chunk.start : chunk.end] == chunk.text for chunk in manifest.chunks)
    for chunk in manifest.chunks:
        page = document.pages[chunk.page_start - 1]
        assert page.start_offset <= chunk.start < chunk.end <= page.end_offset


def test_overlap_retains_content_and_makes_progress() -> None:
    document = _document(("Sentence one. Sentence two. Sentence three. " * 20,))
    manifest = chunk_document(document, ChunkingConfig(180, 40, 80, 40))

    assert len(manifest.chunks) > 2
    pairs = zip(manifest.chunks, manifest.chunks[1:], strict=False)
    assert all(right.start < left.end for left, right in pairs)
    pairs = zip(manifest.chunks, manifest.chunks[1:], strict=False)
    assert all(right.start > left.start for left, right in pairs)


def test_empty_pages_are_ignored_and_table_like_text_is_labeled() -> None:
    document = _document(("", "Column A    Column B\nValue 1     Value 2\nValue 3     Value 4"))
    manifest = chunk_document(document, ChunkingConfig(200, 20, 40, 40))

    assert len(manifest.chunks) == 1
    assert manifest.chunks[0].page_start == 2
    assert manifest.chunks[0].kind is ContentKind.TABLE


def test_heading_is_retained_as_chunk_metadata() -> None:
    document = _document(("PROBABLE CAUSE\n\nThe probable cause was loss of control. " * 10,))
    manifest = chunk_document(document, ChunkingConfig(220, 30, 80, 40))

    assert manifest.chunks[0].section == "PROBABLE CAUSE"


def test_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(target_characters=100, overlap_characters=100)


def test_document_without_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="no non-empty"):
        chunk_document(_document(("",)), ChunkingConfig())


def test_short_boundary_chunk_does_not_advance_one_character_at_a_time() -> None:
    prefix = "A useful sentence. " * 90
    text = prefix + "Probable Cause" + "." * 120 + " 42\n" + ("Next section. " * 90)
    manifest = chunk_document(_document((text,)), ChunkingConfig())

    starts = [chunk.start for chunk in manifest.chunks]
    pairs = zip(starts, starts[1:], strict=False)
    assert all(right - left >= 320 for left, right in pairs)
