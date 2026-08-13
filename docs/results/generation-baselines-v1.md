# Base, prompted, and RAG generation baselines v1

## Decision

RAG with the development-selected hybrid retriever, cross-encoder reranker, and
three context chunks is the first locked generation operating point. It improves
answer overlap and unsupported-question refusal over both context-free variants,
but it is not yet reliable enough to present as a finished aviation QA system.
Retrieval/context selection is the next improvement target before SFT.

## Protocol

- Model: `mistralai/Ministral-3-3B-Instruct-2512`, revision
  `b35d4dfe56c142746f54dbd64f579faab2744308`
- Runtime: Torch 2.7.1+cu128 on the detected RTX 4070 Laptop GPU
- Development: 20 questions and 60 generations
- Locked test: 10 questions and 30 generations
- Decoding: greedy, maximum 48 new tokens
- RAG: development-selected E5/BM25 RRF, MiniLM reranker, top three chunks
- Raw outputs and human-review packets: ignored under `artifacts/evaluation/`
- Tracked metrics contain aggregates only, avoiding publication of test answers.

The development configuration was selected and reviewed before creating
`generation_test_v1.toml`. The test file is bound to SHA-256
`c095f7bd656f0dfebfdd7ea4cb221c7a11d76c7a6aef1df3ebd9cf952e044723`;
the runner refuses it without `--allow-frozen-test`.

## Results

| Split | Variant | Token F1 | Gold evidence coverage | Citation precision | Mean generation latency |
|---|---|---:|---:|---:|---:|
| Development | Base | 0.118 | n/a | n/a | 12.86 s |
| Development | Prompted | 0.237 | n/a | n/a | 10.28 s |
| Development | RAG | 0.335 | 0.778 | 0.778 | 9.10 s |
| Locked test | Base | 0.091 | n/a | n/a | 14.51 s |
| Locked test | Prompted | 0.213 | n/a | n/a | 6.44 s |
| Locked test | RAG | 0.283 | 0.600 | 0.700 | 7.81 s |

Exact match was zero for all variants. It is retained as a diagnostic but rejected
as the primary metric because correct cited answers naturally paraphrase references.
Latency is single-stream generation in one process and excludes retrieval/model
loading; it is not a serving benchmark.

## Development review and error analysis

A six-output RAG spot check covered correct fact answers, a numerical paraphrase,
an unsupported question, and the weakest answer/retrieval cases. It found:

- Correct cited answers for fatalities and aircraft identity.
- “About 200 ft” was correct despite low token F1 against “About 200 feet.”
- Both unsupported questions were correctly refused by RAG; prompted refused one
  of two, while base refused neither.
- Three of the weakest answerable RAG results coincided with failure to retrieve a
  gold evidence chunk.
- One maintenance answer cited related evidence but answered a different omission,
  demonstrating that syntactically valid citations do not prove entailment.

## Locked-test failure analysis

The test run was scored once without parameter changes. RAG improved aggregate
token F1 but retrieved gold evidence for only 60% of questions. The most serious
failure answered the fatalities question using a chunk from another accident,
producing 15 instead of 67 deaths. Other low-F1 outputs were semantically correct
paraphrases, including the recorded 266-ft altitude and Route 4 separation answer.

This separates two future tasks: improve event-aware retrieval/context filtering,
then add human-reviewed semantic correctness and citation-entailment labels. Test
outputs must not be used to select those changes or to construct SFT examples.

## Reproduction

The development command is in the repository README. The locked command is the
same with `generation_test_v1.toml`, `retrieval_test_v1.json`, test output paths,
and the required `--allow-frozen-test` flag. Runs checkpoint every completed
variant and resume only when dataset, corpus, configuration, and model identities
match exactly.
