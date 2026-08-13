# AeroLLM Lab

AeroLLM Lab implements a provenance-preserving NTSB report pipeline and a measured grounded-QA baseline. Version 0.1 acquires and verifies official reports, preserves page-level citations through parsing and chunking, freezes event-family splits, runs lexical/dense/hybrid retrieval with reranking, and compares base, prompted, and retrieval-augmented generation (RAG) with a pinned local Ministral model.

This is a portfolio research release, not an operational aviation assistant.

## Version 0.1 outcomes

- Built NTSB pilot corpus v1: 16 reports, 2,017 pages, and 6,897 chunks.
- Froze 10 independently reviewed v1 test questions from one held-out event family.
- Selected hybrid retrieval and a cross-encoder reranker using development data, then performed one locked test measurement.
- Evaluated base, prompted, and RAG generation with `mistralai/Ministral-3-3B-Instruct-2512` at revision `b35d4dfe56c142746f54dbd64f579faab2744308`.
- Froze the NTSB provenance selection and event-family split for corpus v2, then produced a 72-slot review workbook. Its 27 development examples are drafts awaiting review; its 45 test slots are intentionally unauthored for independent human authoring.

### Measured generation results

| Split | Variant | Token F1 | Gold evidence coverage | Citation precision |
|---|---|---:|---:|---:|
| Development (20 questions) | Base | 0.118 | n/a | n/a |
| Development (20 questions) | Prompted | 0.237 | n/a | n/a |
| Development (20 questions) | RAG | 0.335 | 0.778 | 0.778 |
| Locked test (10 questions) | Base | 0.091 | n/a | n/a |
| Locked test (10 questions) | Prompted | 0.213 | n/a | n/a |
| Locked test (10 questions) | RAG | 0.283 | 0.600 | 0.700 |

Exact match was zero for every variant. RAG is the strongest measured generation baseline, but the locked test exposed a serious retrieval-provenance failure: a fatalities question received context from another accident and the model answered 15 instead of 67. Citation syntax therefore must not be confused with entailment or correct event identity. See the [full result](docs/results/generation-baselines-v1.md).

The locked retrieval results are similarly mixed: hybrid RRF improved Recall@10 from BM25's 0.500 to 0.700, while reranking improved early ordering (MRR@10 0.483, nDCG@10 0.513) but reduced Recall@10 to 0.600 because a gold chunk fell outside the top-20 candidate pool.

### Model and hardware context

The pinned model is Ministral 3 3B Instruct 2512 FP8. Development and all checked-in measurements used an RTX 4070 Laptop GPU with 8,188 MiB VRAM and a 45 W power limit. On that system the one-prompt warm smoke run used 4.96 GB peak VRAM and generated 1.12 token/s.

## Exact quick start

Prerequisites: Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```powershell
uv sync --extra dev
uv run aerollm-doctor
```

For the pinned CUDA/Transformers path:

```powershell
uv sync --extra dev --extra transformers
uv run aerollm-doctor --transformers
```

Validate the frozen v1 test set against a locally materialized pilot corpus:

```powershell
uv run aerollm-freeze-retrieval-test
uv run aerollm-validate-corpus artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_test_v1.json
```

Large source snapshots, corpora, indexes, model weights, and raw runs are ignored local artifacts; cloning alone does not recreate them. Acquisition requires an NTSB Developer Portal key configured in [Data sources](docs/data-sources.md).

## Architecture

```text
official NTSB snapshot -> verified PDF -> page-aware document -> stable chunks
                                                        |
                    event-family split -> corpus/index -> retrieve -> rerank
                                                        |                 |
                                     frozen examples -> generation -> evaluation
```

Immutable hashes connect sources, documents, chunks, corpora, examples, indexes, model revisions, and run configurations. Development data selects configurations; locked test data is measured once and excluded from training and tuning. See [Architecture](docs/architecture.md).

## Status

| Capability | v0.1 status |
|---|---|
| NTSB acquisition, PDF validation, parsing, chunking | Implemented |
| Corpus v1 and independently reviewed test v1 | Frozen and measured |
| BM25, dense, hybrid, cross-encoder reranking | Implemented and measured |
| Base, prompted, and RAG generation | Implemented and measured |
| Corpus v2 provenance selection and event-family split | Frozen |
| Suite v2 | 27 development drafts await review; 45 test slots unauthored |
| QLoRA / SFT and SFT + RAG | Not complete |
| vLLM serving and serving benchmarks | Not complete |
| Tool calling / bounded agents | Not complete |
| DPO / preference tuning | Not complete |
| Educational Transformer implementation | Not complete |

## Documentation

- [v0.1 release notes](docs/release/v0.1.0.md)
- [Release checklist](docs/release/checklist.md)
- [Project status](docs/progress.md)
- [Results index](docs/results/README.md)
- [Hardware profile](docs/hardware-profile.md)
- [Roadmap](docs/roadmap.md)

## Limitations

- The locked v1 test set has only 10 answerable questions from one event family.
- Retrieval can cross event boundaries; the observed wrong-accident answer makes RAG unsuitable for safety-sensitive use.
- Token F1 under-rewards valid paraphrases, while citation precision does not establish that a citation entails an answer.
- All checked-in GPU results come from one low-power RTX 4070 Laptop system.
- Suite v2 is not an evaluation set until independent review and test authoring are complete.
- QLoRA, vLLM, tool calling, DPO, and the educational Transformer remain planned work, not release outcomes.

## License

Code is released under the [MIT License](LICENSE). NTSB source documents retain their own provenance and usage context; large source artifacts are not redistributed here.
