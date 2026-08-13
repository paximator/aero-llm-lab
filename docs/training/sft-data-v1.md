# SFT data v1 validation stage

The first post-training dataset contains 50 deterministic, evidence-linked records
derived only from the 13 event families assigned to the v1 corpus `train` split.
It is a pipeline-validation dataset, not the future 250-record reviewed SFT set.

Each record retains:

- record, event, event-family, report, and source-chunk identities;
- system, user, and assistant messages;
- task type and synthetic provenance;
- review status and an inspectable token-count estimate;
- a `grounded-json-v1` answer with an exact normalized citation span.

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

The 50 records teach grounded extraction and output-schema adherence by selecting a
coherent sentence from a supplied report excerpt. They do not yet provide diverse,
human-authored numeric, causal, abstention, or multi-evidence instructions. This is
appropriate for the QLoRA memory and tiny-overfit gates, but insufficient for the
first meaningful SFT quality claim.

Five distributed records received a model-assisted qualitative sample review in
`data/training/sft_v1_50.sample_review.json`. That review found the records usable
for pipeline validation and identified one lower-value document cross-reference.
It is explicitly not a substitute for repository-owner sampling before training.
