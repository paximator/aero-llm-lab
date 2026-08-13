# Evaluation suite v2 proposal review guide

## Decision

The repository owner approved all proposed targets and policies without
modification on 2026-08-13. The approved plan authorizes corpus-v2 candidate
selection and subsequent blank annotation-packet generation. It does not freeze an
evaluation dataset and does not authorize inspecting future test labels for tuning.

## What you are reviewing

Review `configs/evaluation/evaluation_suite_v2.proposal.toml`. It is a coverage and
governance proposal, not an annotation dataset and not a frozen benchmark.

The proposal deliberately contains no invented new NTSB event IDs. The validated
v1 corpus has only one test family and two development families, so corpus v2 must
add untouched events before a credible multi-family test packet can be authored.

## Decisions requested

Please review these choices first:

1. Overall target: 72 examples; minimum freeze gate: 50.
2. At least eight represented event families, including five test families.
3. No more than 12 examples from one event family.
4. Primary-task allocation: numeric 10, date/time 8, entity/identifier 12,
   categorical/yes-no 8, causal 12, multi-evidence 10, unanswerable 12.
5. Published v1 event families retain their existing splits.
6. New test events remain untouched until their questions are independently
   authored/reviewed and the suite is frozen.

## How to record feedback

Edit only the proposal values and explanatory comments. Do not add private API
keys, raw reports, or test answers to the proposal. For a disputed decision, add a
comment immediately above the value using this form:

```toml
# REVIEW: Prefer 60 because ...
target_total_examples = 72
```

You may also answer the six questions under `[review_questions]` in a separate
message. After approval, the next implementation will acquire/select corpus-v2
events and generate the actual blank annotation packet.

## Supporting documents

- `docs/roadmap.md`: priority and exit gates.
- `docs/task-contract.md`: grounded-QA behavior and leakage invariants.
- `docs/evaluation/retrieval-annotation-guide.md`: current evidence authoring rules.
- `docs/data-sources.md`: source acquisition and credential handling.
- `docs/results/retrieval-test-v1.md`: limitations of the existing frozen suite.
- `docs/results/generation-baselines-v1.md`: failures motivating broader evaluation.

## Approval gate — passed

The proposal is approved only when target size, event-family minimums, task quotas,
unanswerable proportion, split-preservation policy, and table/OCR scope have an
explicit decision. Approval does not freeze the suite; it authorizes corpus-v2
selection and annotation-packet generation.

Recorded decision: target size, event-family minimums, task quotas, unanswerable
proportion, v1 split preservation, and the proposed table/OCR treatment were all
approved without changes.
