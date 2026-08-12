# Data source strategy

## Initial source: NTSB

The first vertical slice uses US National Transportation Safety Board aviation
investigation data. NTSB provides an official developer portal, CAROL search
exports, and downloadable aviation datasets. This gives the project both a
programmatic path and a bulk fallback.

Official references:

- [NTSB Developer Portal](https://developer.ntsb.gov/)
- [NTSB accident data and downloadable datasets](https://www.ntsb.gov/safety/data/pages/Data_Stats.aspx)
- [CAROL help and JSON/CSV export](https://www.ntsb.gov/Pages/carol.aspx)

The legacy downloadable dataset is expected to transition to the Enterprise API in
2027. Consequently, provider-specific payloads must never leak into downstream
code. The source adapter maps remote records into stable internal records.

## Acquisition policy

1. Call the official source through a narrow adapter.
2. Save the unmodified response body and a metadata sidecar atomically.
3. Record request parameters, retrieval time, response headers, and SHA-256 digest.
4. Exclude credentials and authorization headers from all artifacts and logs.
5. Parse only immutable snapshots; never train or evaluate directly from a live call.
6. Pin each dataset build to a source manifest containing snapshot digests.

### Bounded acquisition command

The initial command intentionally acquires exactly one API response for an inclusive
date range of at most 31 days. It does not paginate, normalize cases, or build a
dataset; those operations must read the resulting immutable snapshot.

```powershell
$env:AEROLLM_NTSB_BASE_URL = "https://<endpoint-from-the-developer-portal>"
$env:AEROLLM_NTSB_API_KEY = "<subscription-key>"
uv run aerollm-acquire-ntsb --start 2026-01-01 --end 2026-01-07
```

The base URL and subscription key are required environment variables. The command
sends the key only in the `Ocp-Apim-Subscription-Key` request header, prints a
provider-neutral `SourceDocument`, and writes the raw body plus a metadata sidecar
under the ignored `artifacts/` directory. Request credentials and non-allowlisted
response headers are never persisted.

### Snapshot normalization and manifests

`normalize_snapshot` accepts a provider-neutral `SourceDocument`, verifies the
saved bytes against its SHA-256 digest, and then maps supported NTSB response
envelopes into `AviationReportRecord` values. It has no HTTP dependency and rejects
invalid JSON, unsupported envelopes, missing case identities, and malformed dates.
Only a small allowlist of normalized attributes crosses the provider boundary.

`SourceManifest` records the source, creation time, snapshot identities, artifact
paths, retrieval times, and digests required to reproduce a dataset build. It does
not serialize arbitrary snapshot metadata, keeping credentials and provider-only
response details outside manifests.

This separates convenience from reproducibility: an API can refresh the corpus,
but a historical experiment always resolves to the exact bytes it consumed.

## Secondary source: AAIB

The UK AAIB report collection is a good later extension because its publication
page states Open Government Licence v3.0 coverage except where otherwise noted.
Per-document third-party rights still require review. AAIB should be introduced
only after the NTSB task and evaluation pipeline are stable, and through its own
adapter rather than NTSB-specific parsing.

- [AAIB publications and licensing notice](https://www.gov.uk/government/publications/aaib-publications/aaib-publications)
- [AAIB report search](https://www.gov.uk/aaib-reports)

French BEA material remains a candidate for a multilingual phase. It should not be
mixed into the initial English benchmark before language-specific evaluation and
usage review exist.
