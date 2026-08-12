# Results index

Tracked result pages summarize reviewed conclusions. Their headline values are
generated from versioned machine-readable metrics and ignored immutable run
artifacts; they should not be edited independently of those records.

| Milestone | Status | Primary result | Report |
|---|---|---|---|
| Data pipeline v1 | Complete with source limitations | 16 reports, 2,017 pages, 6,897 chunks | [Report](data-pipeline-v1.md) |
| Extraction audit v1 | Complete | Table label quarantined; header-only chunks removed | [Report](extraction-audit-v1.md) |
| Retrieval test set v1 | Frozen | 10 independently reviewed test examples | [Report](retrieval-test-v1.md) |
| BM25 development v1 | Draft baseline | Recall@10 0.722; MRR@10 0.276 | [Report](bm25-development-v1.md) |
| BM25 test v1 | Complete locked baseline | Recall@10 0.500; MRR@10 0.500 | [Report](bm25-test-v1.md) |
| Dense + hybrid retrieval v1 | Complete | Hybrid test Recall@10 0.700 | [Report](dense-hybrid-retrieval-v1.md) |
| Reranking v1 | Complete | Test MRR@10 0.483; nDCG@10 0.513 | [Report](reranking-v1.md) |
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
