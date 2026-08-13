# Evaluation suite v2 annotation review

## Review file

Open the ignored local workbook at
`artifacts/private_evaluation/evaluation_suite_v2_test_gold.json`.

The tracked public boundary is under `data/evaluation/v2/`: approved development
records are in `development.json`, aggregate coverage is in
`coverage_manifest.json`, and `test_manifest.json` contains only label-free slot
metadata. Never copy test questions, answers, evidence, or review data into the
public manifest.

The workbook contains 72 quota-controlled slots across eight untouched event
families:

- 27 model-assisted development drafts over three development reports;
- 45 blank test slots over five test reports.

The test slots are intentionally blank because the approved governance requires
locked-test questions to be independently human-authored. Do not copy development
questions or use test content to tune prompts, retrieval, or training.

## Development review

For each item with `split` equal to `development`:

1. Open `report_url` and locate each quoted span on the recorded page.
2. Confirm that the question has one unambiguous interpretation.
3. Correct the question, reference answer, required facts, tags, and evidence when
   needed.
4. For unanswerable items, search the full report and replace the draft search note
   with a concise record of what you checked.
5. Set `review_status` to `approved`, `needs_changes`, or `rejected` and explain
   non-approval in `review_notes`.

Whitespace inside an extracted quote may look irregular because page layout is
preserved. You may normalize it while reviewing; the freeze validator will resolve
the reviewed wording back to the exact chunk span.

## Test authoring

For each item with `split` equal to `test` and status `authoring_required`:

1. Use only its assigned `report_url` and `task_type`.
2. Write the question and reference answer yourself.
3. Record the smallest sufficient exact quote or multiple distinct supporting
   quotes, with their pages.
4. Fill task-specific structured targets and required key facts where applicable.
5. Add your identifier to `author` and leave `review_status` as `draft` for a later,
   independent review.

An unanswerable test item needs an empty evidence list and a concrete full-report
search note. Do not invent a merely plausible absent fact without checking it.

## Coverage guardrails

Do not change event families, splits, or task types casually. The workbook already
matches the approved totals exactly: 10 numeric, 8 date/time, 12 entity/identifier,
8 categorical, 12 causal, 10 multi-evidence, and 12 unanswerable examples.
