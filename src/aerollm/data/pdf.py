"""Page-aware PDF parsing from verified source snapshots."""

from __future__ import annotations

import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

from aerollm.common.schemas import Document, PageSpan, SourceDocument


class PDFParseError(ValueError):
    """A source snapshot cannot be parsed as a supported PDF."""


def parse_pdf_snapshot(source: SourceDocument, *, max_bytes: int = 25 * 1024 * 1024) -> Document:
    """Verify an immutable PDF snapshot and extract layout-aware text by page."""
    path = Path(source.artifact_path)
    body = path.read_bytes()
    if len(body) > max_bytes:
        raise PDFParseError(f"PDF snapshot exceeds the {max_bytes}-byte parsing limit")
    digest = hashlib.sha256(body).hexdigest()
    if digest != source.sha256:
        raise PDFParseError(
            f"snapshot digest mismatch for {path}: expected {source.sha256}, got {digest}"
        )
    if not body.startswith(b"%PDF-"):
        raise PDFParseError("snapshot does not have a PDF signature")
    try:
        reader = PdfReader(io.BytesIO(body), strict=True)
    except (DependencyError, PdfReadError, TypeError, ValueError) as error:
        raise PDFParseError("PDF snapshot could not be parsed") from error
    decrypted_with_empty_password = False
    if reader.is_encrypted:
        try:
            decrypted_with_empty_password = bool(reader.decrypt(""))
        except (DependencyError, PdfReadError, TypeError, ValueError) as error:
            raise PDFParseError("encrypted PDF snapshot could not be opened") from error
        if not decrypted_with_empty_password:
            raise PDFParseError("password-protected PDF snapshots are not supported")
    try:
        page_texts = tuple(
            (page.extract_text(extraction_mode="layout") or "").rstrip() for page in reader.pages
        )
    except (PdfReadError, TypeError, ValueError) as error:
        raise PDFParseError("PDF snapshot could not be parsed") from error
    text, spans = _join_pages(page_texts)
    return Document(
        document_id=f"doc-{source.sha256}",
        source_sha256=source.sha256,
        text=text,
        pages=spans,
        parser=f"pypdf/{version('pypdf')}",
        metadata={
            "page_count": len(spans),
            "extraction_mode": "layout",
            "empty_password_decryption": decrypted_with_empty_password,
        },
    )


def write_parsed_document(document: Document, path: Path) -> str:
    """Write normalized JSON atomically and return its SHA-256 digest."""
    from aerollm.data.snapshots import atomic_write

    content = (json.dumps(document.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    atomic_write(path, content)
    return hashlib.sha256(content).hexdigest()


def _join_pages(page_texts: tuple[str, ...]) -> tuple[str, tuple[PageSpan, ...]]:
    parts: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        if parts:
            parts.append("\n\n")
            offset += 2
        start = offset
        parts.append(page_text)
        offset += len(page_text)
        spans.append(PageSpan(page_number=page_number, start_offset=start, end_offset=offset))
    return "".join(parts), tuple(spans)
