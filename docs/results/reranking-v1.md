# Cross-encoder reranking v1

Status: complete development selection and one locked test measurement.

## Objective

Rerank candidates from the frozen hybrid RRF retriever using a query–passage
cross-encoder. Candidate depth was selected only on development data and committed
before the test set was scored.

## Configuration

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- License: Apache-2.0
- Input length: 512 tokens
- Candidate generator: frozen hybrid
  `hybrid_198904b79b5eebe8155a510169a0586e0133b0dfe66570881fac1f85cf0bf3ea`
- Candidate-depth development grid: 20, 50, 100
- Selected depth: 20
- Frozen code/configuration revision: `2d906fa`
- Runtime: Torch 2.6.0+cu126 on RTX 4070 Laptop GPU

## Development selection

| Candidate depth | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---:|---:|---:|---:|---:|
| 20 | **0.7778** | **0.6209** | **0.6584** | **167.3 ms** |
| 50 | 0.7778 | 0.5921 | 0.6369 | 229.9 ms |
| 100 | 0.7778 | 0.5921 | 0.6369 | 371.5 ms |

Depth 20 dominates the deeper alternatives: equal recall, better ranking metrics,
and lower latency.

## Locked test result

| System | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | **0.5000** | 0.5000 | 0.5000 | 0.5000 | **0.5000** | 0.5000 |
| Hybrid RRF | 0.2000 | 0.4000 | **0.6000** | **0.7000** | 0.3476 | 0.4326 |
| Hybrid + reranker | 0.4000 | **0.6000** | **0.6000** | 0.6000 | 0.4833 | **0.5131** |

The reranker materially improves early ranking over the hybrid: Recall@1 doubles,
MRR rises by 0.1357, and nDCG rises by 0.0805. However, it cannot recover a gold
chunk outside the selected top-20 candidate pool, reducing Recall@10 from 0.70 to
0.60. It nearly restores BM25 MRR while retaining one additional top-10 answer.

Mean locked-test latency at k=10 was 165.1 ms per query, including hybrid candidate
generation, query embedding, and GPU cross-encoder scoring.

## Reproduction and artifacts

```powershell
uv sync --extra dev --extra transformers
```

Pinned configuration is stored in
`configs/retrieval/reranker_minilm_l6_v2.toml`. Raw development selection and locked
test reports remain under ignored `artifacts/runs/reranker-*/` directories.

## Limitations and decision

- Only 18 answerable development and 10 test questions are scored.
- The MS MARCO cross-encoder is general English, not aviation-specific.
- Passage truncation may omit evidence late in long chunks.
- Test results will not be used to change candidate depth or reranker settings.

Decision: retain both operating points. Use hybrid top-100 when recall is paramount;
use cross-encoder reranked top-20 when a short, better-ordered RAG context is needed.
The next milestone is generation evaluation, where development data will determine
the context budget before one locked test run.
