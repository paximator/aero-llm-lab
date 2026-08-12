import pytest

from aerollm.data.ntsb import NTSBRequestError
from aerollm.data.ntsb_reports import NTSBReportSource
from aerollm.data.sources import RemoteResponse, ReportDocumentMetadata


def _report(url: str = "https://data.ntsb.gov/report.pdf") -> ReportDocumentMetadata:
    return ReportDocumentMetadata("case:report", url, "investigation-report")


def test_report_source_applies_transport_bounds() -> None:
    calls = []

    def transport(url, headers, timeout, max_bytes, allowed_hosts):
        calls.append((url, headers, timeout, max_bytes, allowed_hosts))
        return RemoteResponse(b"%PDF-synthetic", "application/pdf", {"ETag": "v1"})

    source = NTSBReportSource(
        allowed_hosts=frozenset({"data.ntsb.gov"}),
        timeout_seconds=7,
        max_bytes=1024,
        transport=transport,
    )
    response = source.fetch_document(_report())
    assert response.body == b"%PDF-synthetic"
    assert calls[0][1]["Accept"] == "application/pdf"
    assert calls[0][2:] == (7, 1024, frozenset({"data.ntsb.gov"}))


@pytest.mark.parametrize(
    "url",
    [
        "http://data.ntsb.gov/report.pdf",
        "https://user:password@data.ntsb.gov/report.pdf",
        "https://data.ntsb.gov:8443/report.pdf",
        "https://untrusted.example/report.pdf",
    ],
)
def test_report_source_rejects_unsafe_urls_before_transport(url) -> None:
    source = NTSBReportSource()
    with pytest.raises(ValueError):
        source.fetch_document(_report(url))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (RemoteResponse(b"not-pdf", "application/pdf"), "not a PDF"),
        (RemoteResponse(b"%PDF-data", "text/html"), "content type"),
        (RemoteResponse(b"%PDF-too-long", "application/pdf"), "exceeds"),
    ],
)
def test_report_source_rejects_invalid_responses(response, message) -> None:
    def transport(*args):
        return response

    source = NTSBReportSource(max_bytes=10, transport=transport)
    with pytest.raises(NTSBRequestError, match=message):
        source.fetch_document(_report())
