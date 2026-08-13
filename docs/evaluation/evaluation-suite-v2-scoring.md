# Evaluation suite v2 task scoring

The development suite uses a deliberately small deterministic scorer set. Targets
are reviewed separately in
`configs/evaluation/evaluation_suite_v2.targets.toml` and materialized into
`data/evaluation/v2/development.json`.

| Strategy | Required annotation | Match rule |
| --- | --- | --- |
| `numeric_exact_or_tolerance_v1` | value, unit, tolerance | Any stated number falls within tolerance. |
| `normalized_datetime_v1` | date, local time, timezone label, approximate flag | Normalized date and time both occur. |
| `normalized_entity_set_v1` | non-empty required entity list | Every normalized entity occurs. |
| `categorical_exact_v1` | label | Normalized exact label match. |
| `required_key_facts_v1` | non-empty key-fact list | Every normalized canonical fact occurs. |
| `abstention_v1` | `answerable=false` | The prediction contains no answer. |

Validation fails when a declared strategy lacks its required target. The scorer
reports aggregate accuracy and accuracy by `task_type`; it does not claim semantic
entailment. Canonical key facts are therefore kept as minimal inspectable phrases.

Validate the public annotations:

```powershell
uv run aerollm-evaluate-tasks-v2 data/evaluation/v2/development.json
```

Score predictions containing `example_id` and nullable `answer` fields:

```powershell
uv run aerollm-evaluate-tasks-v2 data/evaluation/v2/development.json `
  --predictions path/to/predictions.json --output path/to/report.json
```
