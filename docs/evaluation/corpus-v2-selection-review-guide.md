# Corpus v2 event-selection review guide

## File under review

Review `artifacts/pilot/corpus_v2_selection.review.json`. This ignored local file
contains 16 metadata-only NTSB event candidates. No report PDFs were downloaded and
no evaluation questions were drafted during selection.

## How to review each event

For every object under `selections`, inspect:

- `event_family_id` and `occurred_on`;
- `proposed_split`;
- severity, weather, event type, report type, country, and location metadata;
- the official `report_url`;
- whether the event adds useful diversity to aviation QA tasks.

Set `review.status` to exactly one of:

```text
approved
needs_changes
rejected
```

Use `review.notes` to explain requested split changes or rejection. Do not write
questions, answers, or evidence notes yet. Opening the public report URL only to
verify that it exists and identifies the stated event is acceptable; content-level
annotation begins after the split plan is approved.

## Split policy

The proposal contains exactly:

- 8 training event families;
- 3 development event families;
- 5 test event families.

All 16 are excluded from corpus v1. Test events must remain untouched by prompt,
retrieval, training-data, and hyperparameter selection. Changing a split changes
the selection identity and requires the final plan to be regenerated and reviewed.

## Approval gate

The next command will refuse to freeze or download the selection until every event
is approved and the 8/3/5 quotas still hold. Rejected events will be replaced from
the remaining deterministic candidate pool, then returned for another review.

After selection approval, the repository will download and validate PDFs, parse and
chunk them, build corpus v2, and only then generate the question-by-question draft
annotation packet.

If an event is rejected, its replacement is written to a separate one-event follow-up
packet. The original review file is preserved. Review that packet with the same status
values; only the replacement needs another human decision.
