from datetime import date

import pytest

from aerollm.data.ntsb import NTSBRequestError, NTSBSource
from aerollm.data.sources import RemoteResponse


def test_ntsb_source_builds_concrete_date_range_request_without_leaking_key() -> None:
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers, timeout))
        return RemoteResponse(b'{"cases": []}', "application/json", {"ETag": "v1"})

    source = NTSBSource(
        base_url="https://api.example.test/root/",
        api_key="top-secret",
        timeout_seconds=12,
        transport=transport,
    )
    report = next(source.list_reports(date(2026, 1, 2), date(2026, 1, 3)))
    response = source.fetch_report(report)

    url, headers, timeout = calls[0]
    assert url == (
        "https://api.example.test/root/aviation/api/GetCasesByDateRangeV2"
        "?startDate=2026-01-02&endDate=2026-01-03"
    )
    assert headers["Ocp-Apim-Subscription-Key"] == "top-secret"
    assert "top-secret" not in report.source_url
    assert timeout == 12
    assert response.body == b'{"cases": []}'


def test_ntsb_source_rejects_invalid_ranges() -> None:
    source = NTSBSource("https://api.example.test", "key")
    with pytest.raises(ValueError, match="end date"):
        next(source.list_reports(date(2026, 1, 2), date(2026, 1, 1)))


def test_ntsb_request_error_does_not_include_credentials(monkeypatch) -> None:
    def failing_transport(url, headers, timeout):
        raise NTSBRequestError("NTSB API request failed")

    source = NTSBSource("https://api.example.test", "top-secret", transport=failing_transport)
    report = next(source.list_reports(date(2026, 1, 1), date(2026, 1, 1)))
    with pytest.raises(NTSBRequestError) as caught:
        source.fetch_report(report)
    assert "top-secret" not in str(caught.value)
