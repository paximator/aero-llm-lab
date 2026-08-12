# Project progress

This page is the narrative milestone log. Reproducible measurements live in
[`docs/results`](results/README.md); raw run artifacts remain under ignored
`artifacts/` directories.

## Completed

- Defined the evaluation-first architecture, task contract, local hardware budget,
  and snapshot-first source policy.
- Integrated the NTSB API with ignored credentials and immutable raw snapshots.
- Implemented verified PDF acquisition, parsing, deterministic chunking, event-level
  corpus splits, validation, and resumable pilot orchestration.
- Froze data-pipeline v1 from the available official aviation reports.
- Audited extraction and content labels; removed confirmed header-only chunks and
  quarantined the unreliable table heuristic from retrieval features.
- Independently reviewed and froze retrieval test set v1: 10 grounded questions with
  exact corpus evidence and an immutable digest.
- Ran the locked BM25 test baseline without test-driven tuning: Recall@10 and MRR@10
  were both 0.500 on 10 examples.
- Added GPU-backed E5 dense retrieval and development-selected RRF. The locked hybrid
  test run improved Recall@10 from 0.500 to 0.700, with lower MRR than BM25.
- Added development-selected MiniLM cross-encoder reranking. On locked test it raised
  hybrid MRR@10 from 0.348 to 0.483, while top-20 candidate recall limited Recall@10
  to 0.600.
- Proved native-Windows generation feasibility for the pinned official Ministral 3
  3B Instruct FP8 checkpoint: 4.96 GB peak allocated VRAM and 1.12 generated
  tokens/second on the detected 45 W RTX 4070 Laptop GPU after kernel caching.

## Current milestone

Build base, prompted, and RAG generation evaluation using the frozen retrieval
operating points. Select prompts and context budgets using development only.

## Next milestones

1. Base, prompted, and RAG generation comparisons on the frozen task suite.
2. QLoRA SFT on the documented local hardware budget.
