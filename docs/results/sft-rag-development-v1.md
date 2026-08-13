# SFT + RAG development comparison v1

Status: complete negative development result.

## Objective and hypothesis

Measure the complete retrieved-context path on the 27 reviewed development
questions. The hypothesis was that the retained QLoRA v1 adapter would use retrieved
evidence more reliably than the base checkpoint.

## Fixed inputs

- Dataset: `evaluation-suite-v2-development`, 27 reviewed examples, SHA-256
  `0ca9f3ef395c4b6b8e9656fc10910731eb326cfb335b5bc16fa71c7f71098f18`.
- Corpus: three official NTSB PDFs (`CEN16MA036`, `DCA13MA081`, `ERA14MA271`),
  745 chunks; SHA-256
  `16c3f49c468fa2a986e7162ce774edcc3e67daf3424a4746a11dcc417bae2c47`.
- Retrieval: BM25 + local `intfloat/e5-small-v2`, RRF (`lexical_weight=0.25`,
  `candidate_k=100`) then local `cross-encoder/ms-marco-MiniLM-L6-v2`
  (`candidate_k=20`), top 3 supplied to generation.
- Generation: local Ministral-3-3B-Base-2512 in NF4, then the retained QLoRA v1
  adapter with SHA-256
  `6ad104d2b666caa2ee32b4700b984ef089c571b978828f31372786df3b9a4bee`.
- Parent code revision: `86bf64e261a95cc3f93cb67383849c2d8f3a8672`; the runner is committed
  with this report. Run-artifact SHA-256:
  `ca0a571d1b551a573fe0078caeb13cf646aa0d84f6ddfbff64668539ef1c3527`.
- Hardware/runtime: NVIDIA RTX 4070 Laptop GPU (8,188 MiB), Python 3.13.7,
  Torch 2.7.1+cu128, CUDA 12.8.

Before generation, the corpus validator resolved every reviewed evidence quote to
the materialized canonical chunks.

## Exact command

```powershell
uv run --extra transformers --extra training aerollm-compare-sft-rag `
  --corpus artifacts/corpus_v2_dev/corpus.json `
  --dense-index artifacts/corpus_v2_dev/dense-index `
  --dense-model artifacts/models/e5-small-v2 `
  --reranker-model artifacts/models/ms-marco-MiniLM-L6-v2 `
  --model artifacts/models/Ministral-3-3B-Base-2512 `
  --adapter artifacts/training/qlora-sft-50-v1 `
  --output artifacts/evaluation/sft-rag-development-v2.json `
  --batch-size 4 --max-new-tokens 192
```

## Results

| Metric | RAG | SFT + RAG |
|---|---:|---:|
| Gold evidence hit@3 | 63.0% | 63.0% |
| Task accuracy after fail-closed validation | 18.5% | 18.5% |
| Valid grounded output | 0.0% | 0.0% |
| Abstention rate | 100.0% | 100.0% |
| Format-failure rate | 100.0% | 100.0% |
| Mean generation latency | 11.53 s | 12.90 s |

Retrieval found at least one annotated gold chunk for 17 of 27 questions. Both
generation variants nevertheless failed the strict response contract for every
example and were converted to safe abstentions. The adapter provides no measurable
benefit in this complete path.

## Error analysis

The result exposes two independent bottlenecks. Retrieval misses 10 annotated
examples at top 3, including the intentionally unanswerable questions where a gold
hit is not expected. More importantly, generation does not produce contract-valid
grounded output even when retrieval succeeds. This is consistent with the earlier
gold-context experiment: the adapter can improve generation with curated evidence,
but that gain does not transfer to longer, noisy retrieved chunks.

## Limitations

This is a development-set diagnostic, not a frozen-test claim. It covers only three
reports and 27 examples; confidence intervals would not be meaningful. Gold hit@3
requires an exact annotated chunk ID and can undercount alternative supporting
chunks. The comparison does not isolate prompt length, context selection, parsing,
and citation-format failures. Model files and raw reports are local ignored
artifacts; the tracked digests and command make their identity auditable but do not
redistribute them.

## Decision

Retain QLoRA v1 as the gold-context development baseline, but do not claim an
SFT+RAG improvement. The next targeted experiment should classify the raw contract
failures and reduce retrieved context or train explicitly against retrieved-context
formatting. No new training run is justified until that failure is isolated.
