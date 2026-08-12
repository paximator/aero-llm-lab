"""Generate the tracked data-pipeline result from immutable local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aerollm.data.snapshots import atomic_write
from aerollm.evaluation.corpus import CorpusManifest


def collect_metrics(
    corpus_path: Path, build_path: Path, plan_path: Path, failures_path: Path,
    *, code_revision: str,
) -> dict[str, object]:
    if not code_revision.strip():
        raise ValueError("code_revision is required")
    corpus_bytes = corpus_path.read_bytes()
    build = _object(build_path)
    plan = _object(plan_path)
    failures = _object(failures_path)
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    if digest != build.get("corpus_sha256"):
        raise ValueError("corpus digest does not match build manifest")
    validation = build.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("corpus build is not valid")
    planned = plan.get("candidates")
    failed = failures.get("failed")
    if not isinstance(planned, list) or not isinstance(failed, list):
        raise ValueError("invalid plan or failure report")
    lengths = sorted(len(chunk.text) for chunk in corpus.chunks)
    split_counts = Counter(source.split.value for source in corpus.sources)
    kind_counts = Counter(chunk.kind.value for chunk in corpus.chunks)
    return {
        "schema_version": 1,
        "milestone": "data-pipeline-v1",
        "code_revision": code_revision,
        "corpus_version": corpus.version,
        "corpus_sha256": digest,
        "config_fingerprint": build.get("config_fingerprint"),
        "planned_reports": len(planned),
        "materialized_reports": len(corpus.documents),
        "source_availability_percent": round(100 * len(corpus.documents) / len(planned), 1),
        "failed_reports": len(failed),
        "failure_categories": dict(sorted(Counter(item["error"] for item in failed).items())),
        "pages": sum(len(document.pages) for document in corpus.documents),
        "chunks": len(corpus.chunks),
        "chunk_characters": {
            "minimum": lengths[0],
            "median": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "maximum": lengths[-1],
        },
        "content_kinds": dict(sorted(kind_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "validation_errors": len(validation.get("errors", [])),
        "validation_warnings": len(validation.get("warnings", [])),
    }


def render_markdown(metrics: Mapping[str, Any]) -> str:
    chunk = metrics["chunk_characters"]
    failures = metrics["failure_categories"]
    splits = metrics["splits"]
    split_summary = (
        f"{splits.get('train', 0)} / {splits.get('development', 0)} / "
        f"{splits.get('test', 0)}"
    )
    failure_rows = "\n".join(
        f"| `{reason}` | {count} |" for reason, count in failures.items()
    ) or "| None | 0 |"
    return f"""# Data pipeline v1

Status: completed with documented source-availability limitations.

## Objective

Build a provenance-preserving aviation corpus from official NTSB reports, retain
page-addressable evidence, enforce event-level splits, and reject invalid sources
rather than weakening acquisition or parsing safeguards.

## Results

| Metric | Value |
|---|---:|
| Planned aviation reports | {metrics['planned_reports']} |
| Materialized reports | {metrics['materialized_reports']} |
| Source availability | {metrics['source_availability_percent']:.1f}% |
| Pages | {metrics['pages']:,} |
| Chunks | {metrics['chunks']:,} |
| Validation errors | {metrics['validation_errors']} |
| Validation warnings | {metrics['validation_warnings']} |
| Train / development / test reports | {split_summary} |

Chunk length in source characters: minimum {chunk['minimum']}, median
{chunk['median']}, p95 {chunk['p95']}, maximum {chunk['maximum']}. Content labels:
{_inline_counts(metrics['content_kinds'])}.

## Source failures

| Sanitized failure | Reports |
|---|---:|
{failure_rows}

The three failed URLs were not admitted: one required authorization and two served
HTML rather than PDF bytes. The successful subset was built explicitly from its
completed manifests. This is preferable to accepting unverifiable content or
mixing other transportation modes into an aviation corpus.

## Reproduction

```powershell
. .\\scripts\\import-local-env.ps1 -Name ntsb
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19 --dry-run
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19
$manifests = Get-ChildItem artifacts/manifests/pilot -Filter *.json |
  Sort-Object Name | ForEach-Object FullName
uv run aerollm-build-corpus @manifests
uv run aerollm-report-data-pipeline --code-revision {metrics['code_revision']}
```

## Provenance

- Corpus version: `{metrics['corpus_version']}`
- Corpus SHA-256: `{metrics['corpus_sha256']}`
- Corpus configuration: `{metrics['config_fingerprint']}`
- Data-producing code revision: `{metrics['code_revision']}`
- Machine-readable metrics: [`data-pipeline-v1.metrics.json`](data-pipeline-v1.metrics.json)

## Limitations and decision

This is a small pilot dominated by fatal investigations and is not representative
of all aviation events. PDF text extraction quality has not yet received a sampled
human audit. Six tiny chunks remain flagged for retrieval error analysis. The
layout-spacing heuristic labels {metrics['content_kinds'].get('table', 0):,} chunks
as tables; this unexpectedly high share is treated as a suspected classification
error until manual sampling validates or replaces the heuristic.

Decision: freeze source text, offsets, pages, and splits for the first lexical
retrieval experiment, but do not use `kind` as a retrieval feature yet. Create a
hand-authored, evidence-linked evaluation set before tuning retrieval parameters or
adding dense retrieval.
"""


def write_results(metrics: Mapping[str, object], output_dir: Path) -> tuple[Path, Path]:
    metrics_path = output_dir / "data-pipeline-v1.metrics.json"
    report_path = output_dir / "data-pipeline-v1.md"
    atomic_write(
        metrics_path,
        (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode(),
    )
    atomic_write(report_path, render_markdown(metrics).encode())
    return metrics_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render data-pipeline milestone results")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpora/ntsb-pilot-v1.json"))
    parser.add_argument(
        "--build", type=Path,
        default=Path("artifacts/corpora/ntsb-pilot-v1.build.json"),
    )
    parser.add_argument("--plan", type=Path, default=Path("artifacts/pilot/plan.json"))
    parser.add_argument("--failures", type=Path, default=Path("artifacts/pilot/failures.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    arguments = parser.parse_args(argv)
    try:
        metrics = collect_metrics(
            arguments.corpus, arguments.build, arguments.plan, arguments.failures,
            code_revision=arguments.code_revision,
        )
        metrics_path, report_path = write_results(metrics, arguments.output_dir)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"metrics": metrics_path.as_posix(), "report": report_path.as_posix()}))
    return 0


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _percentile(values: list[int], fraction: float) -> int:
    return values[round((len(values) - 1) * fraction)]


def _inline_counts(values: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={count:,}" for name, count in values.items())


if __name__ == "__main__":
    raise SystemExit(main())
