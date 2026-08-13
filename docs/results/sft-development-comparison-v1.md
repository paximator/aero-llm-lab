# SFT development comparison v1

Status: completed controlled generation comparison; retrieval comparison pending.

## Question

Does the first 50-record QLoRA adapter improve grounded generation on the 27
reviewed evaluation-suite-v2 development questions that were excluded from SFT?

This experiment supplies the reviewed gold evidence spans to both variants. It
therefore isolates generation and structured grounding from retrieval. It must not
be described as RAG performance or frozen-test performance.

## Compared systems

- `base_gold_context`: pinned Ministral 3 3B Base, NF4, no adapter.
- `sft_gold_context`: the same loaded model plus adapter
  `sha256:6ad104d2b666caa2ee32b4700b984ef089c571b978828f31372786df3b9a4bee`.

Both use the same explicit `[SYSTEM]/[USER]/[ASSISTANT]` prompt, greedy decoding,
192 maximum new tokens, the same reviewed evidence, and `PostprocessorV1`.
Malformed output fails closed to an abstention. Raw predictions and per-example
warnings are retained under ignored `artifacts/evaluation/sft-gold-context-v2.json`.

## Results

| Metric | Base | SFT | Difference |
|---|---:|---:|---:|
| Task accuracy after fail-closed processing | 18.5% (5/27) | 44.4% (12/27) | +25.9 pp |
| Valid grounded output | 0.0% (0/27) | 37.0% (10/27) | +37.0 pp |
| Format failure | 100.0% (27/27) | 63.0% (17/27) | -37.0 pp |
| Mean latency per example, batched | 2,664 ms | 4,007 ms | +1,343 ms |

The Base score is entirely the five unanswerable cases: every Base output was
invalid JSON and therefore became a safe abstention. It produced no valid grounded
answer. SFT produced ten valid cited JSON answers. These included all three date/time
questions, two of four entity/identifier questions, and three of four numeric
questions at the format layer; deterministic answer scoring accepted two numeric
answers.

The remaining SFT failures were: seven invalid citations, five invalid JSON outputs,
and five malformed schemas. Causal and multi-evidence questions remained at 0/4
correct. Several answers were semantically close but failed exact citation or
required-key-fact checks, which is evidence that the current extraction-only SFT
data does not teach composition across multiple facts.

## Reproduction

```powershell
uv run --extra transformers --extra training aerollm-compare-sft `
  --model artifacts/models/Ministral-3-3B-Base-2512 `
  --adapter artifacts/training/qlora-sft-50-v1 `
  --output artifacts/evaluation/sft-gold-context-v2.json `
  --batch-size 4 --max-new-tokens 192
```

## Interpretation and next gate

This is a positive but bounded result: the adapter transfers format and some
single-span extraction behavior to three unseen event families. It does not yet
support a quality claim for causal or multi-evidence QA, and its format reliability
is too low for production use.

The next data iteration should add reviewed abstention, causal, multi-evidence,
categorical, and exact-citation examples, then repeat this unchanged comparison.
Only after the three development reports are acquired and chunked canonically
should the same Base/SFT comparison be extended to retrieved context as RAG and
SFT+RAG.
