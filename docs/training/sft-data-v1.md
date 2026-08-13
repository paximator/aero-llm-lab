# SFT data v1 validation stage

The first post-training dataset contains 50 deterministic, evidence-linked records
derived only from the 13 event families assigned to the v1 corpus `train` split.
It is a pipeline-validation dataset, not the future 250-record reviewed SFT set.

Each record retains:

- record, event-family, and source-chunk identities;
- system, user, and assistant messages;
- a `grounded-json-v1` answer with an exact normalized citation span.

Fields derivable from those identities or messages are deliberately omitted: report
and event IDs, investigation URL, PDF pages, task type, provenance label, review
status, and token count. The validator reconstructs provenance and token usage from
the canonical corpus instead of trusting duplicated values in the JSON.

The builder balances records across train families, rejects evidence crossing report
boundaries, rejects non-train reports, duplicate IDs/messages, malformed assistant
JSON, unresolved citations, and records exceeding the configured token budget.
Evaluation questions are not reused.

Build from an existing canonical corpus:

```powershell
uv run aerollm-build-sft-data artifacts/corpora/ntsb-pilot-v1.json `
  --output data/training/sft_v1_50.json
```

## Deliberate limitation

The 50 records teach grounded extraction and output-schema adherence by presenting
the selected evidence sentence as the bounded report excerpt. The builder rejects
merged section headings, page/list fragments, broken punctuation, and detectable
OCR word splits. They do not yet provide diverse,
human-authored numeric, causal, abstention, or multi-evidence instructions. This is
appropriate for the QLoRA memory and tiny-overfit gates, but insufficient for the
first meaningful SFT quality claim.

All 50 records received a model-assisted qualitative review after the deterministic
quality gate. The dataset-level `review` object records the scope and conclusion
once, rather than repeating a pending status on every record. The first ten also
retain a detailed review trace in
`data/training/sft_v1_50.sample_review.json`. The review drove stricter deterministic
filters and context bounding. This remains explicitly distinct from repository-owner
approval before training.

Inspect one record without joining corpus internals manually:

```powershell
uv run aerollm-review-sft-data data/training/sft_v1_50.json `
  --record RECORD_ID
```

The command prints the compact provenance identity, dataset review comment,
instruction, expected answer, and exact citation. Join the source chunk to the
canonical corpus when report URL or PDF-page information is needed for a deeper audit.
