# Data pipeline v1

Status: completed with documented source-availability limitations.

## Objective

Build a provenance-preserving aviation corpus from official NTSB reports, retain
page-addressable evidence, enforce event-level splits, and reject invalid sources
rather than weakening acquisition or parsing safeguards.

## Results

| Metric | Value |
|---|---:|
| Planned aviation reports | 19 |
| Materialized reports | 16 |
| Source availability | 84.2% |
| Pages | 2,017 |
| Chunks | 6,897 |
| Validation errors | 0 |
| Validation warnings | 0 |
| Train / development / test reports | 13 / 2 / 1 |

Chunk length in source characters: minimum 103, median
1480, p95 1854, maximum 1996. Content labels:
table=5,679, text=1,218.

## Source failures

| Sanitized failure | Reports |
|---|---:|
| `NTSB report returned HTTP 401` | 1 |
| `NTSB report returned unsupported content type 'text/html'` | 2 |

The three failed URLs were not admitted: one required authorization and two served
HTML rather than PDF bytes. The successful subset was built explicitly from its
completed manifests. This is preferable to accepting unverifiable content or
mixing other transportation modes into an aviation corpus.

## Reproduction

```powershell
. .\scripts\import-local-env.ps1 -Name ntsb
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19 --dry-run
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19
$manifests = Get-ChildItem artifacts/manifests/pilot -Filter *.json |
  Sort-Object Name | ForEach-Object FullName
uv run aerollm-build-corpus @manifests
uv run aerollm-report-data-pipeline --code-revision 9abcd77
```

## Provenance

- Corpus version: `ntsb-pilot-v1`
- Corpus SHA-256: `37aa9ed77c3476b149ed8f4d455a7e8680e9c0359edf323117c422d4a765ac6d`
- Corpus configuration: `sha256:8493da349eae666a43cec42dbd12a6d59bd8ef6fe48bef9dd284527545f93a03`
- Data-producing code revision: `9abcd77`
- Machine-readable metrics: [`data-pipeline-v1.metrics.json`](data-pipeline-v1.metrics.json)

## Limitations and decision

This is a small pilot dominated by fatal investigations and is not representative
of all aviation events. PDF text extraction quality has not yet received a sampled
human audit. An extraction audit removed six confirmed header-only chunks; the
rebuilt corpus has 0 validation warnings. The
layout-spacing heuristic labels 5,679 chunks
as tables; this unexpectedly high share is treated as a suspected classification
error until manual sampling validates or replaces the heuristic.

Decision: freeze source text, offsets, pages, and splits for the first lexical
retrieval experiment, but do not use `kind` as a retrieval feature yet. Create a
hand-authored, evidence-linked evaluation set before tuning retrieval parameters or
adding dense retrieval.
