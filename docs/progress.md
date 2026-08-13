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
- Completed development-selected and digest-locked base/prompted/RAG generation
  baselines. On the frozen test set, token F1 increased from 0.091 to 0.213 to
  0.283; RAG evidence coverage was limited to 0.600 by retrieval.
- Completed and froze the serving-v1 prototype after aligning its versioned JSON
  prompt, chunk-aware context, and fail-closed `PostprocessorV1` behavior with the
  offline grounded pipeline. Advanced serving performance work remains deferred.
- Built and reviewed the first 50-record train-only, evidence-linked SFT validation
  dataset over 13 event families, with leakage, provenance, duplicate, schema,
  citation, and token-budget validation.
- Completed QLoRA memory smoke, tiny-overfit, adapter save/reload, and a first
  50-record run on the RTX 4070 Laptop GPU. The 50-record adapter remains the
  development baseline.
- Compared Base and SFT on all 27 reviewed development questions with identical gold
  evidence. Task accuracy improved from 18.5% to 44.4%, and valid grounded output
  from 0% to 37.0%; this isolates generation and is not a RAG claim.
- Ran a failure-driven 90-record corrective experiment. It regressed accuracy to
  33.3%, exposed and fixed ordered-training recency bias, and was rejected rather
  than promoted.
- Materialized the three evaluation-suite-v2 development reports into a validated
  745-chunk corpus and ran the complete RAG versus SFT+RAG path. Retrieval hit an
  annotated gold chunk at top 3 for 63.0%, but both generators failed closed on all
  examples; this is retained as a negative development result.
- Classified the retrieved-context failures and rejected two five-example fixes:
  Base remains non-JSON, while SFT starts the schema but truncates or cites an
  invalid span. Canonical prompt and full-context defaults remain unchanged.

## Current milestone

Build a small retrieved-context training gate with concise complete JSON, distractor
chunks, and explicit sequence termination before spending time on another full run.

Evaluation-suite v2 currently has 27 approved development examples and 45
ready-to-fill test slots that still require independent human authoring and review.
See the [suite v2 status and remaining work](evaluation/evaluation-suite-v2-status.md)
for the explicit delivery boundary.

## Next milestones

1. Evaluation-suite v2: 50–100 reviewed examples across independent event families.
2. Task-specific scoring and a shared failure taxonomy.
3. Separate prompt-length, citation-format, parsing, and context-selection failures
   in the completed RAG versus SFT+RAG development run.
4. Author genuinely new train-split causal and multi-evidence SFT examples before a
   future v3; templated recomposition is rejected by the v2 result.
5. Complete independent test authoring before any frozen five-system claim.

The detailed priority gates and explicit deferred scope are maintained in
[`docs/roadmap.md`](roadmap.md).
