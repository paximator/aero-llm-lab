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

## Current milestone

Develop a cross-encoder reranker using only the development split. Keep the frozen
hybrid candidate generator unchanged and treat its test result as locked.

## Next milestones

1. Development-selected cross-encoder reranking over frozen hybrid candidates.
2. Base, prompted, and RAG generation comparisons on the frozen task suite.
3. QLoRA SFT on the documented local hardware budget.
