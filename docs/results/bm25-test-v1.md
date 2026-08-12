# BM25 test baseline v1

Status: complete locked test measurement; no test-driven tuning permitted.

## Objective

Measure the already-defined deterministic BM25 v1 retriever on the independently
reviewed retrieval test set. The corpus, dataset, tokenizer, BM25 parameters, and
cutoffs were fixed before observing test metrics.

## Fixed inputs

- Code revision at execution: `b4f89757be14d3582a8189ee3bc31cb885c251c0`
- Corpus SHA-256: `37aa9ed77c3476b149ed8f4d455a7e8680e9c0359edf323117c422d4a765ac6d`
- Dataset SHA-256: `c095f7bd656f0dfebfdd7ea4cb221c7a11d76c7a6aef1df3ebd9cf952e044723`
- Dataset: `ntsb-retrieval-test` version `1.0.0`, 10 answerable examples
- Index: `bm25_e4bca72529b99926e117b422bda9aafd5cbc54aa16352bf71eb1e0b3f460d2a9`
- BM25: `k1=1.2`, `b=0.75`, tokenizer `unicode-lexical-v1`
- Host: Windows 11; latency measured on the local development machine

## Results

| Cutoff | Scored questions | Recall@k | MRR@k | nDCG@k | Mean latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.5000 | 0.5000 | 0.5000 | 12.095 ms |
| 3 | 10 | 0.5000 | 0.5000 | 0.5000 | 11.633 ms |
| 5 | 10 | 0.5000 | 0.5000 | 0.5000 | 11.825 ms |
| 10 | 10 | 0.5000 | 0.5000 | 0.5000 | 12.351 ms |

Five gold chunks ranked first. The other five gold chunks were absent from the top
10, so increasing the cutoff did not change aggregate effectiveness. These values
are a small-sample locked estimate, not evidence that BM25 recall is exactly 50% on
the broader aviation domain.

## Exact commands

The same command was executed with `K` equal to `1`, `3`, `5`, and `10`:

```powershell
uv run aerollm-retrieve artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_test_v1.json --k K `
  --json-output artifacts/runs/bm25-test-v1/report-kK.json `
  --markdown-output artifacts/runs/bm25-test-v1/report-kK.md
```

Raw per-query results remain under ignored `artifacts/runs/bm25-test-v1/`. Tracked
headline values are stored in `bm25-test-v1.metrics.json`.

## Error observation and leakage boundary

The locked run found five examples at rank 1 and missed five within rank 10. The
missed set covered summary facts, a probable-cause phrase, and a qualification
finding. This test observation is recorded only to characterize baseline behavior.
It must not be used to change tokenization, query wording, BM25 parameters, chunking,
or reranking. Detailed retrieval development and error-driven changes use only the
development split.

## Limitations

- Ten questions from one event family produce high sampling uncertainty.
- All examples are answerable; this run does not measure abstention.
- Latency is a single local run, not a warmed, repeated systems benchmark.
- Chunk recall does not independently score whether both spans inside a multi-span
  gold chunk were localized after retrieval.

## Decision

Retain BM25 v1 unchanged as the lexical test baseline. Develop dense retrieval using
only development data, select its frozen configuration there, and then run one locked
test comparison against this result.
