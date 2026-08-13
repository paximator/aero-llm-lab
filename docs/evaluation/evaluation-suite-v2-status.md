# Evaluation suite v2 status

This note records the implementation status and the deliberate delivery boundary
of evaluation suite v2 for development review. It must not be read as a frozen
benchmark announcement.

## Status at a glance

| Area | Status | Meaning |
| --- | --- | --- |
| Coverage plan | Approved | The target is 72 examples with fixed task-type quotas. |
| Corpus and split assignment | Ready | Eight independent event families are assigned: three to development and five to test. |
| Development examples | Complete for this stage | 27 model-assisted question/answer records have been reviewed and approved. |
| Test examples | Public structure ready; private content outstanding | 45 quota-controlled slots expose only their report, event family, task type, and scoring strategy; labels require independent human authoring in an ignored local workbook. |
| Suite freeze | Not ready | The suite remains `2.0.0-draft` with status `human_review_and_test_authoring_required`. |

## Deliberate scope choice

The initial implementation completed 27 development question/answer records to
exercise the annotation schema and workflow end to end: task allocation, exact
evidence resolution, answerability, scoring strategy, tags, review status, and
coverage accounting.

The remaining 45 records were intentionally created as ready-to-fill test slots.
Their stable fields are already populated, but their questions, reference answers,
evidence, structured targets, required facts, authors, and reviews must be supplied
through the human workflow. This time-boxed boundary makes the unfinished work
visible and avoids presenting model-generated test labels as independent human
evaluation data.

The public/private boundary is now enforced in the repository. Public development
data and label-free test metadata live under `data/evaluation/v2/`; the complete
blank authoring template is published there as a pedagogical record of the intended
workflow. The filled test-gold workbook belongs under the ignored
`artifacts/private_evaluation/` path. Future test gold must remain outside Git.

This is therefore a completed development scaffold, not a completed evaluation
suite. The 27 approved development records can support workflow development and
diagnostics. The 45 test slots cannot yet support benchmark results or model-quality
claims.

## Methodological position

- Published v1 event families keep their historical splits.
- Test event families remain isolated from prompt, retrieval, training, and
  hyperparameter selection.
- Model-assisted drafting is allowed for development records and is recorded with
  `is_synthetic: true`.
- Test questions must be independently human-authored; later review must be carried
  out by someone other than the author.
- Coverage is controlled before test authoring so that annotation effort cannot
  silently reshape the benchmark around convenient questions.
- The suite is frozen only after schema, evidence, duplicate, leakage, coverage,
  and independent-review gates pass.

## Remaining work

1. Human-author all 45 test slots from their assigned reports and task types.
2. Complete task-specific targets, required key facts, evidence spans, and
   full-report search notes for unanswerable examples.
3. Independently review every test record and resolve requested changes.
4. Run schema, exact-evidence, event-family leakage, duplicate/near-duplicate,
   task-target consistency, and coverage validation.
5. Freeze the final bytes and SHA-256 digest, then record the migration boundary
   from the historical v1 benchmark.

Until those steps are complete, the correct top-level status is
`human_review_and_test_authoring_required`.

## Review entry points

- Coverage decision: `configs/evaluation/evaluation_suite_v2.proposal.toml`
- Public development set: `data/evaluation/v2/development.json`
- Public label-free test manifest: `data/evaluation/v2/test_manifest.json`
- Public blank authoring template: `data/evaluation/v2/test_authoring_template.json`
- Private local workbook: `artifacts/private_evaluation/evaluation_suite_v2_test_gold.json`
- Annotation procedure: `docs/evaluation/evaluation-suite-v2-annotation-review-guide.md`
- Corpus selection procedure: `docs/evaluation/corpus-v2-selection-review-guide.md`
