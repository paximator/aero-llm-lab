"""Bounded retrieval of NTSB report documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from aerollm.data.ntsb import NTSBRequestError
from aerollm.data.sources import RemoteResponse, ReportDocumentMetadata

PDFTransport = Callable[[str, Mapping[str, str], float, int, frozenset[str]], RemoteResponse]


@dataclass(frozen=True, slots=True)
class NTSBReportSource:
    """Fetch official PDF documents through an allowlisted, size-bounded boundary."""

    allowed_hosts: frozenset[str] = frozenset({"www.ntsb.gov", "data.ntsb.gov"})
    timeout_seconds: float = 30.0
    max_bytes: int = 25 * 1024 * 1024
    transport: PDFTransport | None = None

    @property
    def name(self) -> str:
        return "ntsb"

    def __post_init__(self) -> None:
        if not self.allowed_hosts:
            raise ValueError("at least one allowed host is required")
        if self.timeout_seconds <= 0 or self.max_bytes <= 0:
            raise ValueError("timeout_seconds and max_bytes must be positive")
        object.__setattr__(
            self, "allowed_hosts", frozenset(host.casefold() for host in self.allowed_hosts)
        )

    def fetch_document(self, document: ReportDocumentMetadata) -> RemoteResponse:
        self._validate_url(document.source_url)
        transport = self.transport or _urlopen_pdf_transport
        response = transport(
            document.source_url,
            {
                "Accept": "application/pdf",
                "User-Agent": "Mozilla/5.0 (compatible; aerollm-lab/0.1)",
            },
            self.timeout_seconds,
            self.max_bytes,
            self.allowed_hosts,
        )
        _validate_pdf_response(response, self.max_bytes)
        return response

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("report URL must use HTTPS")
        if parsed.username or parsed.password:
            raise ValueError("report URL must not contain credentials")
        if parsed.port not in {None, 443}:
            raise ValueError("report URL must use the default HTTPS port")
        if (parsed.hostname or "").casefold() not in self.allowed_hosts:
            raise ValueError("report URL host is not allowlisted")


def _validate_pdf_response(response: RemoteResponse, max_bytes: int) -> None:
    if len(response.body) > max_bytes:
        raise NTSBRequestError(f"NTSB report exceeds the {max_bytes}-byte limit")
    media_type = response.content_type.partition(";")[0].strip().casefold()
    if media_type not in {"application/pdf", "application/octet-stream"}:
        raise NTSBRequestError(f"NTSB report returned unsupported content type {media_type!r}")
    if not response.body.startswith(b"%PDF-"):
        raise NTSBRequestError("NTSB report response is not a PDF")


def _urlopen_pdf_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_bytes: int,
    allowed_hosts: frozenset[str],
) -> RemoteResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 (validated URL)
            final_host = (urlparse(response.geturl()).hostname or "").casefold()
            if final_host not in allowed_hosts:
                raise NTSBRequestError("NTSB report redirected to a non-allowlisted host")
            body = response.read(max_bytes + 1)
            response_headers = dict(response.headers.items())
            content_type = response.headers.get_content_type()
    except HTTPError as error:
        raise NTSBRequestError(f"NTSB report returned HTTP {error.code}") from error
    except (URLError, TimeoutError) as error:
        raise NTSBRequestError("NTSB report request failed") from error
    return RemoteResponse(body=body, content_type=content_type, headers=response_headers)
