import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerollm.common.schemas import SourceDocument
from aerollm.data.pdf import PDFParseError, parse_pdf_snapshot, write_parsed_document

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ntsb" / "synthetic-report.pdf.base64"


def _pdf_bytes() -> bytes:
    return base64.b64decode(FIXTURE.read_text(encoding="ascii"))


def _source(path, body: bytes, digest: str | None = None) -> SourceDocument:
    path.write_bytes(body)
    return SourceDocument(
        source="ntsb",
        source_id="CEN26FA001:report",
        source_url="https://data.ntsb.gov/report.pdf",
        publisher="NTSB",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        sha256=digest or hashlib.sha256(body).hexdigest(),
        artifact_path=path.as_posix(),
    )


def test_pdf_parser_preserves_page_text_and_offsets(tmp_path) -> None:
    parsed = parse_pdf_snapshot(_source(tmp_path / "report.bin", _pdf_bytes()))
    assert "Synthetic NTSB report page one." in parsed.text
    assert len(parsed.pages) == 1
    page = parsed.pages[0]
    assert parsed.text[page.start_offset : page.end_offset] == parsed.text
    assert parsed.metadata == {
        "page_count": 1,
        "extraction_mode": "layout",
        "empty_password_decryption": False,
    }

    output = tmp_path / "parsed" / "report.json"
    digest = write_parsed_document(parsed, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


def test_document_allows_parser_separators_between_page_spans() -> None:
    from aerollm.common.schemas import Document, PageSpan

    digest = "a" * 64
    document = Document(
        document_id=f"doc-{digest}",
        source_sha256=digest,
        text="page one\n\npage two",
        pages=(PageSpan(1, 0, 8), PageSpan(2, 10, 18)),
        parser="fixture",
    )

    second = document.pages[1]
    assert document.text[second.start_offset : second.end_offset] == "page two"


@pytest.mark.parametrize(
    ("body", "digest", "message"),
    [
        (b"not-pdf", None, "PDF signature"),
        (_pdf_bytes(), "0" * 64, "digest mismatch"),
    ],
)
def test_pdf_parser_rejects_unverified_input(tmp_path, body, digest, message) -> None:
    source = _source(tmp_path / "report.bin", body, digest)
    with pytest.raises(PDFParseError, match=message):
        parse_pdf_snapshot(source)


def test_crypto_dependency_failure_is_a_controlled_parse_error(tmp_path, monkeypatch) -> None:
    source = _source(tmp_path / "report.bin", b"%PDF-synthetic")

    def fail_reader(*args, **kwargs):
        from pypdf.errors import DependencyError

        raise DependencyError("missing crypto provider")

    monkeypatch.setattr("aerollm.data.pdf.PdfReader", fail_reader)
    with pytest.raises(PDFParseError, match="could not be parsed"):
        parse_pdf_snapshot(source)
