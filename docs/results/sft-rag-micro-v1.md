# Retrieved-context micro-SFT v1

Status: complete negative gate; no larger run authorized.

## Objective and gate

Test whether retrieved-context examples and an explicit EOS training target remove
the dominant SFT+RAG truncation failure. Continue only if at least 4 of the same 5
development questions produce valid grounded output.

## Fixed inputs

- Parent: reviewed, train-only SFT v1 dataset.
- Micro-dataset: 12 records (8 answerable, 4 abstentions), each with one selected
  reviewed context and two train-only distractors.
- Micro-dataset SHA-256:
  `d304e64baca846e778b50fa71f622c8623349e280c2c055328db4178c3cc11e7`.
- Training: NF4 QLoRA, rank 16, alpha 32, learning rate `2e-4`, seed 42,
  60 shuffled steps, maximum sequence length 1,024, explicit EOS target.
- Adapter SHA-256:
  `2f7ff79cdc91f73ad42327009b305bdc91ba9ca7619e6f493d5212404cee2948`.
- Evaluation: first five reviewed development questions, retrieval hit@3 5/5,
  concise-v2 prompt, selected passages, greedy generation capped at 256 tokens.
- Evaluation artifact SHA-256:
  `00507dacdb16606724030b967492646df3b60089e7abf7d427fff3d061c89b78`.

## Exact commands

```powershell
uv run aerollm-build-retrieved-context-sft data/training/sft_v1_50.json `
  --output data/training/sft_rag_micro_v1_12.json

uv run --extra transformers --extra training aerollm-train-qlora `
  --model artifacts/models/Ministral-3-3B-Base-2512 `
  --dataset data/training/sft_rag_micro_v1_12.json `
  --output artifacts/training/qlora-sft-rag-micro-v1 `
  --records 12 --steps 60

uv run --extra transformers --extra training aerollm-compare-sft-rag `
  --corpus artifacts/corpus_v2_dev/corpus.json `
  --dense-index artifacts/corpus_v2_dev/dense-index `
  --dense-model artifacts/models/e5-small-v2 `
  --reranker-model artifacts/models/ms-marco-MiniLM-L6-v2 `
  --model artifacts/models/Ministral-3-3B-Base-2512 `
  --adapter artifacts/training/qlora-sft-rag-micro-v1 `
  --output artifacts/evaluation/sft-rag-micro-v1-mini5.json `
  --batch-size 1 --max-new-tokens 256 --limit 5 `
  --prompt-version concise-v2 --context-mode passage
```

## Results

Training loss fell from `0.084004` to `0.000097`. The adapter reloaded successfully,
used 4,442 MiB peak allocated VRAM, and trained in 273.4 seconds.

| Mini-gate metric | Result |
|---|---:|
| Valid grounded SFT+RAG outputs | 0/5 |
| Correct SFT+RAG answers after fail-closed validation | 0/5 |
| Generation truncations | 0/5 |
| Incorrect citations | 3/5 |
| Schema violations | 1/5 |
| Unexpected valid abstentions | 1/5 |

The EOS-aware micro-training fixed the previous truncation behavior: completions
ended after 33–195 tokens instead of reaching the cap. It did not pass the grounding
gate. Three outputs selected plausible answers and correct chunk IDs but altered
the exact quote through punctuation or spacing; the strict postprocessor correctly
rejected them. One answerable question produced a structurally valid abstention,
and one encoded an abstention with a non-null answer.

## Limitations

This is deliberate overfitting on 12 automatically recomposed records, not a model
quality experiment. The questions are templated and distractors derive from the
reviewed parent dataset rather than a live retriever. Five development questions
are only a rejection gate. The adapter is retained as an ignored diagnostic artifact
and is not promoted over QLoRA v1.

## Decision

The 4/5 gate failed, so stop before producing 50 records or running all 27 questions.
The next data iteration would need citation-copy supervision that preserves exact
source spans and a better balance between answerable and abstention behavior. This
requires reviewed data authoring, not merely more steps on these 12 records.
