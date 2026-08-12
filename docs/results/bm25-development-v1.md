# BM25 development baseline v1

Status: untuned development baseline; not a frozen test result.

## Objective

Measure lexical retrieval over the frozen NTSB pilot corpus before changing BM25
parameters, adding embeddings, or introducing reranking.

## Fixed inputs

- Corpus SHA-256: `37aa9ed77c3476b149ed8f4d455a7e8680e9c0359edf323117c422d4a765ac6d`
- Development dataset SHA-256: `a294b0c38b89058b9447f9ccd076ae487c4636944161e1e6f7b69d13f0fdcdd1`
- Questions: 20 total; 18 answerable and 2 negative
- Index: dependency-free deterministic BM25 defaults
- Content-type labels were not used as retrieval features

## Results

| Cutoff | Scored questions | Recall@k | MRR@k | nDCG@k | Mean latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 0.1111 | 0.1111 | 0.1111 | 12.116 ms |
| 5 | 18 | 0.5000 | 0.2454 | 0.3085 | 12.160 ms |
| 10 | 18 | 0.7222 | 0.2756 | 0.3809 | 12.286 ms |

The large gain between ranks 1 and 10 shows that lexical matching often finds the
evidence but ranks competing passages from long, repetitive investigation reports
too highly. This establishes a useful baseline for query analysis and reranking.

## Reproduction

```powershell
uv run aerollm-build-retrieval-dataset
uv run aerollm-retrieve artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_dev_v1.json --k 10 `
  --json-output artifacts/runs/bm25-dev-v1/report-k10.json `
  --markdown-output artifacts/runs/bm25-dev-v1/report-k10.md
```

## Limitations and decision

The questions are model-assisted development annotations and have not received
independent human review. Latency is a small local single-query measurement, not a
systems benchmark. The test packet remains unscored pending human review. Metrics
must therefore be treated as engineering diagnostics, not portfolio headline test
claims.

Decision: manually review a sample of development questions, complete independent
review of the 10 test drafts, and categorize BM25 misses before changing retrieval.

