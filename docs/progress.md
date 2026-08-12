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

## Current milestone

Develop and select the first dense retriever using only the development split. The
locked BM25 test result is recorded for final comparison and must not drive changes.

## Next milestones

1. Dense retrieval development and measured comparison with BM25.
2. One locked dense-retrieval test measurement after configuration selection.
3. Reranking only if baseline failures justify it.
4. Base, prompted, and RAG generation comparisons on the frozen task suite.
5. QLoRA SFT on the documented local hardware budget.
