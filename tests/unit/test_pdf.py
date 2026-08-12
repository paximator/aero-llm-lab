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
    assert parsed.metadata == {"page_count": 1, "extraction_mode": "layout"}

    output = tmp_path / "parsed" / "report.json"
    digest = write_parsed_document(parsed, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


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
