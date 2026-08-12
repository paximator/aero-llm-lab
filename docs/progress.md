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

## Current milestone

Create the evidence-linked retrieval evaluation set, then run the first BM25
baseline over the frozen corpus without tuning against the test split.

## Next milestones

1. BM25 retrieval evaluation and error analysis.
2. Dense retrieval and measured comparison with BM25.
3. Reranking only if baseline failures justify it.
4. Base, prompted, and RAG generation comparisons on the frozen task suite.
5. QLoRA SFT on the documented local hardware budget.

