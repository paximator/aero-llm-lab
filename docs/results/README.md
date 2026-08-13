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
| Generation baselines v1 | Complete | Test token F1: base 0.091, prompted 0.213, RAG 0.283 | [Report](generation-baselines-v1.md) |
| QLoRA SFT v1 | Complete development baseline | 50 records; reloadable adapter; 4,375 MiB peak VRAM | [Report](qlora-sft-v1.md) |
| SFT development comparison v1 | Complete, gold-context only | Accuracy 18.5% Base → 44.4% SFT; valid grounded output 0% → 37.0% | [Report](sft-development-comparison-v1.md) |
| Corrective SFT v2 | Complete negative result | Accuracy regressed to 33.3%; v1 retained | [Report](sft-corrective-v2.md) |
| SFT + RAG development v1 | Complete negative result | Hit@3 63.0%; both variants 18.5% accuracy and 100% fail-closed | [Report](sft-rag-development-v1.md) |
| Generation smoke v1 | Complete feasibility baseline | 4.96 GB peak VRAM; 1.12 token/s warm | [Report](generation-smoke-v1.md) |
| Inference benchmark v1 | Pending | — | — |

Every report must identify its dataset/corpus digest, code revision, configuration,
exact command, hardware when relevant, metrics with units, failures, limitations,
and the decision supported by the result.

Start new reports from [`template.md`](template.md). Prefer a deterministic renderer
that consumes `metrics.json`; review and interpret the generated result rather than
copying numeric values into prose by hand.
