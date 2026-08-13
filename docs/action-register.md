# Development action register

This register turns the roadmap into reviewable work items. Update the status and
evidence link when an item changes; do not mark work complete from implementation
alone when human review or a measured gate is part of its acceptance criterion.

## Active work

### A1 — Complete evaluation-suite v2 test gold

- Status: `NEXT`
- Priority: P0
- Expected owner: human author plus a different human reviewer
- Inputs: 45 public label-free slots, assigned reports, annotation guide, private
  ignored workbook
- Work: author questions and targets, resolve exact evidence, document full-report
  searches for unanswerable cases, then independently review every record
- Done when: all required gold fields are complete, every test record has an
  independent reviewer, and requested changes are resolved
- Evidence: private completed workbook and public aggregate review status; never
  commit test gold
- Estimate: the dominant remaining effort; schedule as human annotation sessions,
  not an automated coding task

### A3 — Author retrieved-context training records

- Status: `NEXT` in parallel with A1
- Priority: P0
- Expected owner: dataset author plus reviewer
- Inputs: train-split reports only, failure report `sft-rag-micro-v1`, exact-citation
  contract
- Work: write answerable and abstention examples containing realistic distractors;
  require exact source-span copying and concise complete JSON
- Done when: records are reviewed, traceable to immutable train chunks, balanced by
  failure category, within token budget, and free of test-event contamination
- Evidence: tracked dataset manifest, review record, validator report, and digest
- Do not do: scale the automatically recomposed 12-record diagnostic dataset

## Blocked and gated work

### A2 — Freeze evaluation-suite v2

- Status: `BLOCKED` by A1
- Expected owner: evaluation maintainer
- Work: run schema, exact-evidence, split leakage, duplicate/near-duplicate,
  task-target, review, and coverage validation
- Done when: validation is clean and the final immutable dataset digest plus v1→v2
  migration boundary are recorded
- Failure behavior: return individual records to author/reviewer; never weaken a
  validator to make the freeze pass

### A4 — Pass the SFT+RAG development gate

- Status: `GATED` by A3
- Expected owner: ML engineer
- Work: tiny-overfit, reload and fingerprint the adapter, then evaluate the same five
  development questions with fixed retrieval/prompt settings
- Pass: at least 4/5 outputs are schema-valid, exactly cited, and grounded
- Fail: record categories and stop; do not run 27 examples or enlarge training
- Evidence: dataset, adapter and run digests; loss trace; per-example predictions;
  gate decision report

### A5 — Frozen five-system comparison

- Status: `BLOCKED` by A2 and A4
- Expected owner: evaluation maintainer
- Systems: minimal-instruct, prompted, RAG, SFT, SFT+RAG
- Done when: all systems use the same frozen suite and prediction contract, with
  paired results, uncertainty, task slices, failure categories and limitations
- Prohibited: prompt, retrieval or training choices informed by frozen-test labels

## Deferred work

### A6 — Secondary systems work

- Status: `DEFERRED`
- Includes: streaming, vLLM evaluation, DPO, agents, distributed serving and
  production-scale observability
- Reopen only when: a measured P0 result or explicit deployment requirement gives
  the work a concrete acceptance criterion

## Weekly update template

```text
Action: A#
Status: NEXT | IN_PROGRESS | BLOCKED | GATED | DONE | DEFERRED
Evidence added:
Gate result:
New blocker:
Next owner/action:
```
