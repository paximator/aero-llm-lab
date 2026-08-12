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

## Current milestone

Review the evidence-linked test packet, then freeze and run the first test retrieval
benchmark without tuning against the test split. The untuned development BM25
baseline is recorded for diagnostics.

## Next milestones

1. BM25 retrieval evaluation and error analysis.
2. Dense retrieval and measured comparison with BM25.
3. Reranking only if baseline failures justify it.
4. Base, prompted, and RAG generation comparisons on the frozen task suite.
5. QLoRA SFT on the documented local hardware budget.
