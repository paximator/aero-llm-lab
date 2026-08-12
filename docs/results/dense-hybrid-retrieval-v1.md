# Dense and hybrid retrieval v1

Status: complete development selection and one locked test measurement.

## Objective

Measure semantic retrieval over the frozen NTSB corpus, then test whether reciprocal
rank fusion (RRF) improves the BM25 baseline. All model and fusion selection used the
development split. The chosen hybrid configuration was committed before test scoring.

## Fixed inputs

- Corpus SHA-256: `37aa9ed77c3476b149ed8f4d455a7e8680e9c0359edf323117c422d4a765ac6d`
- Development dataset SHA-256: `a294b0c38b89058b9447f9ccd076ae487c4636944161e1e6f7b69d13f0fdcdd1`
- Test dataset SHA-256: `c095f7bd656f0dfebfdd7ea4cb221c7a11d76c7a6aef1df3ebd9cf952e044723`
- Encoder: `intfloat/e5-small-v2` revision
  `f9611f088d69fc3157ff1878217feee72bda0145`
- Encoding: 512 tokens, attention-mask mean pooling, L2 normalization,
  `query:` and `passage:` prefixes
- Dense search: exact cosine over 6,897 vectors of 384 float32 values
- Dense index: 10,593,792 bytes; SHA-256
  `21e505c91806f16f3de25db6f7396081ba5546dc69ad8e4ba5400834ffad85ec`
- Runtime: Torch `2.6.0+cu126`, CUDA 12.6, RTX 4070 Laptop GPU

## Development selection

| Retriever | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 v1 | 0.7222 | 0.2756 | 0.3809 |
| E5-small-v2 dense | 0.6667 | 0.4265 | 0.4836 |
| Hybrid RRF | 0.7222 | **0.4710** | **0.5295** |

Dense retrieval improved ranking quality but did not beat BM25 Recall@10. RRF was
therefore selected. Lexical weights `0.25`, `0.50`, and `0.75` were compared on
development; all reached Recall@10 `0.7222`, while `0.25` produced the best MRR and
nDCG. The frozen hybrid uses lexical weight `0.25`, dense weight `0.75`, RRF constant
`60`, and 100 candidates from each retriever.

## Locked test comparison

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 v1 | **0.5000** | **0.5000** | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| Hybrid RRF | 0.2000 | 0.4000 | **0.6000** | **0.7000** | 0.3476 | 0.4326 |

Hybrid retrieval increases test Recall@10 by 20 percentage points and finds two
additional gold chunks, but it pushes some easy BM25 hits down the ranking. It is
therefore better for high-recall RAG candidate collection, not an across-the-board
ranking win. A reranker is justified; its design and selection must use development
data only.

## Performance

GPU embedding of all 6,897 chunks plus model loading completed in about 50 seconds.
The persisted embedding matrix is about 10.1 MiB. Mean hybrid query latency in the
locked k=10 test run was 111.2 ms. This implementation performs exact scoring with
Python loops, so latency is an engineering baseline rather than an optimized result.

## Reproduction

```powershell
uv sync --extra dev --extra transformers

uv run aerollm-dense-retrieve artifacts/corpora/ntsb-pilot-v1.json `
  artifacts/indexes/e5-small-v2 artifacts/models/intfloat-e5-small-v2 `
  --model-id intfloat/e5-small-v2 `
  --model-revision f9611f088d69fc3157ff1878217feee72bda0145 `
  --device cuda --batch-size 32 --build
```

Model weights, persistent indexes, selection runs, and per-query reports remain under
ignored `artifacts/`. Tracked aggregate metrics are stored beside this report.

## Limitations and decision

- The development and test sets are small, so changes of one query are material.
- Exact Python-loop cosine search understates achievable serving throughput.
- The encoder is English-only and truncates passages at 512 tokens.
- Test outcomes were observed once and will not be used to retune fusion.

Decision: retain the frozen hybrid as the high-recall retrieval baseline. Next,
implement a development-selected cross-encoder reranker and compare it without
altering the locked hybrid candidate generator.
