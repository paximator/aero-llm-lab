"""Concrete adapter for the NTSB Enterprise Aviation API."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from aerollm.data.sources import RemoteResponse, ReportMetadata

Transport = Callable[[str, Mapping[str, str], float], RemoteResponse]


class NTSBRequestError(RuntimeError):
    """A sanitized error raised when the NTSB request fails."""


@dataclass(frozen=True, slots=True)
class NTSBSource:
    """Retrieve raw date-range payloads without exposing NTSB shapes downstream."""

    base_url: str
    api_key: str
    endpoint_path: str = "aviation/api/GetCasesByDateRangeV2"
    timeout_seconds: float = 30.0
    transport: Transport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def name(self) -> str:
        return "ntsb"

    def list_reports(self, start: date, end: date) -> Iterator[ReportMetadata]:
        """Describe one raw bulk response; case-level parsing happens from its snapshot later."""
        if end < start:
            raise ValueError("end date must be on or after start date")
        parameters = {"startDate": start.isoformat(), "endDate": end.isoformat()}
        url = self._request_url(parameters)
        yield ReportMetadata(
            source_id=f"cases-{start.isoformat()}-{end.isoformat()}",
            source_url=url,
            attributes=parameters,
        )

    def fetch_report(self, report: ReportMetadata) -> RemoteResponse:
        parameters = {
            key: report.attributes[key]
            for key in ("startDate", "endDate")
            if key in report.attributes
        }
        if len(parameters) != 2:
            raise ValueError("NTSB report metadata lacks date-range parameters")
        transport = self.transport or _urlopen_transport
        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "aerollm-lab/0.1",
        }
        return transport(self._request_url(parameters), headers, self.timeout_seconds)

    def _request_url(self, parameters: Mapping[str, str]) -> str:
        base = self.base_url.rstrip("/") + "/"
        endpoint = urljoin(base, self.endpoint_path.lstrip("/"))
        return f"{endpoint}?{urlencode(parameters)}"


def _urlopen_transport(url: str, headers: Mapping[str, str], timeout: float) -> RemoteResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 (configured URL)
            body = response.read()
            response_headers = dict(response.headers.items())
            content_type = response.headers.get_content_type()
    except HTTPError as error:
        raise NTSBRequestError(f"NTSB API returned HTTP {error.code}") from error
    except (URLError, TimeoutError) as error:
        raise NTSBRequestError("NTSB API request failed") from error
    return RemoteResponse(body=body, content_type=content_type, headers=response_headers)
