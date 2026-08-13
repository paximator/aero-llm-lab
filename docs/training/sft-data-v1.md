# SFT data v1 validation stage

The first post-training dataset contains 50 deterministic, evidence-linked records
derived only from the 13 event families assigned to the v1 corpus `train` split.
It is a pipeline-validation dataset, not the future 250-record reviewed SFT set.

Each record retains:

- record, event, event-family, report, and source-chunk identities;
- the official NTSB investigation page and exact PDF page numbers;
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

The 50 records teach grounded extraction and output-schema adherence by presenting
the selected evidence sentence as the bounded report excerpt. The builder rejects
merged section headings, page/list fragments, broken punctuation, and detectable
OCR word splits. They do not yet provide diverse,
human-authored numeric, causal, abstention, or multi-evidence instructions. This is
appropriate for the QLoRA memory and tiny-overfit gates, but insufficient for the
first meaningful SFT quality claim.

The first ten records received a model-assisted qualitative pre-review in
`data/training/sft_v1_50.sample_review.json`. The review drove stricter deterministic
filters and context bounding, then accepted nine records and retained one valid but
lower-value document cross-reference with an explicit caveat.
It is explicitly not a substitute for repository-owner sampling before training.

Inspect one record without joining corpus internals manually:

```powershell
uv run aerollm-review-sft-data data/training/sft_v1_50.json `
  --record RECORD_ID
```

The command prints the official investigation page, the PDF page number(s), the
source chunk, instruction, expected answer, and exact citation. On the NTSB page,
open the `Reports` section to reach the associated PDF. The dataset deliberately
labels this as an investigation URL rather than claiming it is a direct PDF URL.
