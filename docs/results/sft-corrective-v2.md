# Corrective SFT v2 experiment

Status: completed negative result; adapter rejected as the new baseline.

## Hypothesis

The v1 comparison showed 17/27 fail-closed outputs and no correct causal or
multi-evidence answers. A 40-record corrective extension was built from train-only
facts: ten causal, ten two-citation, ten exact-citation, and ten structured
abstention examples. The extension reuses only the 50 previously reviewed facts and
does not use development questions or test events.

## Training correction discovered during the experiment

The first v2 run traversed records in file order. Because abstentions were the final
ten records of every epoch, the adapter learned to abstain on all 27 development
questions. This exposed an order/recency bug in the small-data runner.

The runner now performs a deterministic shuffle for every epoch. The corrected run
used the same model, NF4/LoRA configuration, seed, 90 records, and 270 steps. Its
adapter digest is
`sha256:1a760751a9bb5a43c1df0af281f55f3901f0e3129a2994e89a69f133ded96501`.
Loss fell from `0.479269` to `0.0000425`; peak VRAM was 4,647.46 MiB. Save and reload
passed.

## Unchanged development comparison

| Metric | SFT v1 (50) | Corrective v2 (90) | Change |
|---|---:|---:|---:|
| Task accuracy after fail-closed | 44.4% | 33.3% | -11.1 pp |
| Valid grounded output | 37.0% | 22.2% | -14.8 pp |
| Abstention rate | 63.0% | 77.8% | +14.8 pp |
| Format failure rate | 63.0% | 59.3% | -3.7 pp |

V2 retained 3/3 date/time answers but scored 0/4 numeric, 0/4 causal, and 0/4
multi-evidence. It modestly reduced format failures while making the model too
conservative and reducing answer correctness.

## Decision

Reject the corrective v2 adapter as a replacement for v1. Keep the 50-record v1
adapter as the current development baseline. More epochs on the corrective data are
not justified because training loss is already near zero.

The failed experiment indicates that templated recomposition of the same facts does
not provide the semantic diversity required for causal and multi-evidence transfer.
A future v3 must use genuinely authored train-split questions with task-specific
answers and negative examples calibrated against answerable cases. Its acceptance
gate remains the unchanged 27-example development comparison.
