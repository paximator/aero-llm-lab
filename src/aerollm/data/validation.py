"""Quality checks for generated retrieval corpora."""

from __future__ import annotations

from dataclasses import dataclass

from aerollm.evaluation.corpus import CorpusManifest


@dataclass(frozen=True, slots=True)
class CorpusValidationReport:
    documents: int
    chunks: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "documents": self.documents,
            "chunks": self.chunks,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_built_corpus(
    corpus: CorpusManifest, *, minimum_chunk_characters: int
) -> CorpusValidationReport:
    if minimum_chunk_characters < 1:
        raise ValueError("minimum_chunk_characters must be positive")
    errors: list[str] = []
    warnings: list[str] = []
    chunks_by_document: dict[str, list[object]] = {}
    for chunk in corpus.chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        if len(chunk.text.strip()) < minimum_chunk_characters:
            warnings.append(f"tiny chunk: {chunk.chunk_id}")

    text_digests: dict[str, str] = {}
    for document in corpus.documents:
        previous = text_digests.setdefault(document.source_sha256, document.document_id)
        if previous != document.document_id:
            errors.append(f"duplicate source content: {document.document_id}")
        chunks = chunks_by_document.get(document.document_id, [])
        covered = {
            page.page_number
            for page in document.pages
            if document.text[page.start_offset : page.end_offset].strip()
            and any(
                chunk.start < page.end_offset and chunk.end > page.start_offset
                for chunk in chunks
            )
        }
        expected = {
            page.page_number
            for page in document.pages
            if document.text[page.start_offset : page.end_offset].strip()
        }
        for page_number in sorted(expected - covered):
            errors.append(f"missing chunks for {document.document_id} page {page_number}")

    return CorpusValidationReport(
        len(corpus.documents), len(corpus.chunks), tuple(errors), tuple(warnings)
    )

