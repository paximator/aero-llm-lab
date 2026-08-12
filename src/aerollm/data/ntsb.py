"""Concrete adapter for the NTSB Enterprise Aviation API."""

from __future__ import annotations

import hashlib
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
    endpoint_path: str = "Common/v2/GetCasesByDateRange/"
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

    def list_reports(
        self,
        start: date,
        end: date,
        *,
        mode: str | None = None,
        marker: str | None = None,
    ) -> Iterator[ReportMetadata]:
        """Describe one raw bulk response; case-level parsing happens from its snapshot later."""
        if end < start:
            raise ValueError("end date must be on or after start date")
        parameters = {"startDate": start.isoformat(), "endDate": end.isoformat()}
        if mode:
            parameters["mode"] = mode
        if marker:
            parameters["marker"] = marker
        url = self._request_url(parameters)
        page = f"-marker-{hashlib.sha256(marker.encode()).hexdigest()[:12]}" if marker else ""
        yield ReportMetadata(
            source_id=f"cases-{start.isoformat()}-{end.isoformat()}{page}",
            source_url=url,
            attributes=parameters,
        )

    def fetch_report(self, report: ReportMetadata) -> RemoteResponse:
        parameters = {
            key: report.attributes[key]
            for key in ("startDate", "endDate", "mode", "marker")
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

    def fetch_reference(self, endpoint_path: str) -> RemoteResponse:
        """Fetch a no-parameter reference resource such as version or dictionary."""
        if not endpoint_path or endpoint_path.startswith(("http://", "https://")):
            raise ValueError("reference endpoint must be a relative path")
        transport = self.transport or _urlopen_transport
        headers = {
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "aerollm-lab/0.1",
        }
        base = self.base_url.rstrip("/") + "/"
        return transport(urljoin(base, endpoint_path.lstrip("/")), headers, self.timeout_seconds)

    def fetch_aviation_case(
        self, endpoint_path: str, *, ntsb_number: str, mkey: int | None = None
    ) -> tuple[str, RemoteResponse]:
        """Fetch one aviation case detail by its discovery-record identity."""
        number = ntsb_number.strip()
        if not number or not number.replace("-", "").isalnum():
            raise ValueError("ntsb_number must be a non-empty aviation case identifier")
        if mkey is not None and (type(mkey) is not int or mkey <= 0):
            raise ValueError("mkey must be a positive integer when provided")
        if not endpoint_path or endpoint_path.startswith(("http://", "https://")):
            raise ValueError("aviation case endpoint must be a relative path")
        parameters: dict[str, str | int] = {"ntsbNumber": number}
        if mkey is not None:
            parameters["mkey"] = mkey
        base = self.base_url.rstrip("/") + "/"
        endpoint = urljoin(base, endpoint_path.lstrip("/"))
        url = f"{endpoint}?{urlencode(parameters)}"
        headers = {
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "aerollm-lab/0.1",
        }
        transport = self.transport or _urlopen_transport
        return url, transport(url, headers, self.timeout_seconds)

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
