# Results index

Tracked result pages summarize reviewed conclusions. Their headline values are
generated from versioned machine-readable metrics and ignored immutable run
artifacts; they should not be edited independently of those records.

| Milestone | Status | Primary result | Report |
|---|---|---|---|
| Data pipeline v1 | Complete with source limitations | 16 reports, 2,017 pages, 6,903 chunks | [Report](data-pipeline-v1.md) |
| BM25 retrieval v1 | Pending | — | — |
| Dense retrieval v1 | Pending | — | — |
| Reranking v1 | Pending | — | — |
| Generation baselines v1 | Pending | — | — |
| QLoRA SFT v1 | Pending | — | — |
| SFT + RAG v1 | Pending | — | — |
| Inference benchmark v1 | Pending | — | — |

Every report must identify its dataset/corpus digest, code revision, configuration,
exact command, hardware when relevant, metrics with units, failures, limitations,
and the decision supported by the result.

Start new reports from [`template.md`](template.md). Prefer a deterministic renderer
that consumes `metrics.json`; review and interpret the generated result rather than
copying numeric values into prose by hand.
