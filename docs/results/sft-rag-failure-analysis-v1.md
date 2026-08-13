# SFT + RAG failure analysis v1

Status: complete negative development diagnosis.

## Objective

Explain the 100% fail-closed rate in the first retrieved-context comparison, test
the smallest plausible prompt/context corrections on five examples, and avoid a
full rerun unless the mini-gate produces contract-valid answers.

## Diagnosis

All 54 original predictions carried `invalid_json`. The failure modes were not the
same:

- Base RAG: 0/27 outputs started with JSON; two attempted fenced JSON. The primary
  category is `non_json_output`.
- SFT+RAG: 27/27 outputs started with `{`, but all 27 ended with unmatched braces.
  The adapter learned the schema prefix, then copied long retrieved passages until
  the generation cap. The primary category is `generation_truncation`.

The runner now records prompt/completion token counts and assigns a stable primary
category without relaxing the fail-closed postprocessor.

## Mini-gates

Both gates used the first five development questions, where retrieval gold hit@3
was 100%, `max_new_tokens=256`, and the same model, adapter, index, and reranker as
the original run.

| Gate | Base RAG failures | SFT+RAG failures | Valid output |
|---|---|---|---:|
| Explicit concise schema, full chunks | 5 non-JSON | 4 truncations, 1 incorrect citation | 0/10 |
| Explicit concise schema, selected passages | 5 non-JSON | 3 truncations, 2 incorrect citations | 0/10 |

The selected-passage gate reduced mean latency from 41.85 s to 11.36 s for Base RAG
and from 48.79 s to 25.39 s for SFT+RAG in batch-size-one runs. This timing is only
diagnostic. It did not improve the validity gate.

Ignored run artifact digests:

- Full chunks: `886333f08063e6c3c21b2bd229ac8ad7e76e6c80d6c10522992a0ba602a7ee36`.
- Selected passages: `51e1584b4b3343342769b2f75a94718c7a424ccae65e1d0fceadd6fd81fa34c7`.

Representative reproduction options are now explicit rather than silently changing
the canonical runner:

```powershell
aerollm-compare-sft-rag ... --limit 5 --max-new-tokens 256 `
  --prompt-version concise-v2 --context-mode full

aerollm-compare-sft-rag ... --limit 5 --max-new-tokens 256 `
  --prompt-version concise-v2 --context-mode passage
```

## Limitations

Five development examples are sufficient only as a rejection gate. Passage
selection is lexical and can omit supporting context. The original legacy artifact
does not contain token counts, so its truncation diagnosis uses unmatched JSON plus
the known generation cap; new artifacts record the counts directly.

## Decision

Reject both prompt/context variants for promotion and do not spend GPU time on a
27-example rerun. Keep canonical defaults at prompt `v1` and full chunks. The next
credible fix is training-data work: teach concise, complete JSON using retrieved and
distractor-bearing contexts, with an EOS-aware objective and a tiny-overfit gate,
before evaluating another adapter.
