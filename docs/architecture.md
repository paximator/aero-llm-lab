# Repository architecture

## Scope

The first vertical slice should support one defensible workflow end to end:

> Given a public aviation safety report, answer a factual question or produce a
> constrained summary, cite the supporting passages, and abstain when the report
> does not support the answer.

This narrow scope exercises ingestion, parsing, retrieval, generation, grounding,
evaluation, and serving. Additional tasks—taxonomy classification, causal-factor
extraction, cross-report synthesis, and tool-assisted analysis—can be added after
the slice is reliable.

Safety reports are historical evidence, not operational flight guidance. The API
and demo should state this boundary clearly.

## Target layout

```text
aero-llm-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── .env.example
├── configs/
│   ├── data/                 # source and preprocessing configurations
│   ├── eval/                 # suites, judges, thresholds, slices
│   ├── retrieval/            # chunking, embedding, reranking
│   ├── training/             # SFT and optional preference recipes
│   └── serving/              # engine and API configurations
├── data/
│   ├── README.md             # provenance, licenses, acquisition instructions
│   ├── manifests/            # versioned source metadata and checksums
│   └── samples/              # tiny redistributable fixtures only
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   ├── data-card.md
│   ├── model-card.md
│   └── decisions/            # short architecture decision records
├── src/aerollm/
│   ├── data/                 # acquire, parse, normalize, deduplicate, split
│   ├── retrieval/            # chunk, index, retrieve, rerank
│   ├── generation/           # prompts and inference adapters
│   ├── training/             # dataset formatting and PEFT entry points
│   ├── evaluation/           # task runners, metrics, slices, reports
│   ├── agents/               # typed tools and bounded orchestration
│   ├── serving/              # FastAPI schemas and vLLM integration
│   ├── benchmarking/         # latency, throughput, quality/cost measurements
│   └── common/               # schemas, IDs, hashing, logging, reproducibility
├── transformer_lab/          # educational implementation, isolated from product
│   ├── model.py              # attention, MLP, RMSNorm, residual blocks
│   ├── rope.py
│   ├── cache.py
│   ├── generate.py
│   └── README.md             # equations and parity demonstrations
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── evals/
│   ├── datasets/             # versioned human-reviewed evaluation records
│   ├── rubrics/              # explicit grading and abstention rules
│   └── baselines/            # checked-in aggregate results, not large outputs
├── scripts/                  # thin operator entry points; logic stays in src/
├── notebooks/               # exploration only, never canonical pipelines
├── reports/                  # generated experiment summaries and plots
└── infra/                    # containers and deployment manifests when needed
```

Large reports, model weights, embeddings, indexes, and run outputs do not belong in
Git. Git stores manifests, hashes, small fixtures, configurations, and summarized
results. An artifact store or local ignored directory holds the large objects.

## System boundaries

```text
public reports
     │
     ▼
acquire → parse/OCR → normalize → deduplicate → document-level split
     │                                      │
     │                                      ├── evaluation examples (frozen)
     │                                      └── training examples
     ▼
chunk → embed → index → retrieve → rerank ───────────────┐
                                                        ▼
question + prompt ────────────────────────────────► generation
                                                        │
                              citations + answer + trace│
                                                        ▼
                    deterministic metrics + human rubric + judge audit
```

Training consumes only the training partition. Retrieval indexes are built per
experiment: a production-style corpus index may contain train and non-evaluation
documents, while a closed-book report QA evaluation can explicitly provide the
target report. The protocol must state which setting is being measured.

## Core contracts

Use stable typed records across subsystems rather than passing ad hoc dictionaries.
Suggested conceptual schemas are:

- `SourceDocument`: source URL, publisher, publication date, license/usage note,
  retrieval timestamp, checksum, report/event ID, and local artifact reference.
- `Document`: canonical text plus page/section spans and parser metadata.
- `Chunk`: stable chunk ID, document ID, text, structural location, and offsets.
- `Example`: task, input, expected answer or rubric, evidence spans, split, author,
  and provenance.
- `RetrievalResult`: query, ranked chunks, component scores, index version, timing.
- `GenerationTrace`: model and adapter IDs, prompt version, decoding parameters,
  retrieved context, tool calls, output, token counts, timings, and seed.
- `EvaluationResult`: example ID, system variant, metric values, failure labels,
  evaluator versions, and trace reference.

Every derived artifact should have a content-derived or deterministic identifier.
An experiment manifest ties together the Git commit, configuration, source
manifest, dataset version, split version, model revision, adapter, index, and
environment.

## Evaluation design

Evaluation is a first-class package, not the final notebook. The initial suite
should contain a small, manually verified gold set with these task families:

| Task | Primary measures | Important slices |
|---|---|---|
| Structured extraction | field F1, exact match | missing fields, tables, OCR |
| Report-grounded QA | answer correctness, citation precision/recall | answerable vs unanswerable |
| Constrained summary | factual coverage, unsupported-claim rate | long reports, multiple causes |
| Retrieval | Recall@k, MRR/nDCG, latency | lexical, semantic, identifier queries |
| Tool use (later) | tool selection, argument validity, task success | tool failure, ambiguous input |

LLM-as-judge scores must be calibrated against a human-reviewed subset and reported
separately from deterministic metrics. The suite should record paired outputs so
comparisons can use bootstrap confidence intervals and per-example regressions,
not only mean scores.

Leakage controls are mandatory: split by event/report family, deduplicate before
splitting, track source dates, keep synthetic data out of evaluation, and inspect
near-duplicate overlap. Because a pretrained model may already have seen public
reports, results should distinguish domain adaptation from memorization and favor
questions that require supplied evidence and citations.

## Retrieval and generation

Begin with transparent baselines: lexical retrieval and dense retrieval, followed
by hybrid fusion. Add a cross-encoder or model-based reranker only after retrieval
error analysis shows room for it. Preserve page/section metadata through chunking
so citations can be verified.

All five principal system variants should implement one inference interface. This
keeps prompts, decoding, traces, and evaluation comparable. RAG and fine-tuning are
orthogonal configuration choices rather than separate applications.

## Training

SFT examples should be produced through a reviewable pipeline and retain links to
their evidence. Start with LoRA, use QLoRA when memory measurements justify it, and
compare against the prompted baseline before expanding the dataset. Record trainable
parameter count, token distribution, packing policy, optimizer state, precision,
hardware, wall time, peak VRAM, and seeds.

Preference tuning is a gated experiment. It begins only when preference pairs have
a clear rubric and SFT failure analysis identifies behavior that DPO can plausibly
improve. It is not part of the minimum viable project.

## Agent and tool boundary

Agentic behavior should solve bounded aviation-analysis tasks, not add an open-ended
chat loop. Candidate typed, read-only tools include report lookup, passage search,
metadata filtering, unit conversion, and a small statistics calculator. Validate
arguments, cap iterations, expose tool traces, simulate failures in tests, and
measure task success against a non-agentic baseline.

## Serving and benchmarking

FastAPI owns transport, validation, health/readiness, and trace IDs. vLLM owns
batched model execution. Keep an engine abstraction so local tests can use a small
or mocked backend.

Benchmark quality and systems behavior together. At minimum report time to first
token, inter-token latency, end-to-end latency percentiles, tokens/second, requests/
second, peak allocated/reserved VRAM, prompt/output lengths, concurrency, batch
policy, quantization, hardware, and model revision. Separate cold start from warm
runs and publish the exact load shape.

## Educational Transformer

`transformer_lab/` is intentionally independent of the production stack. It should
implement causal multi-head attention, RoPE, RMSNorm, residual/MLP blocks, autoregressive
generation, and a preallocated KV cache. Tests should compare cached and uncached
logits, verify causal masking and RoPE properties, and demonstrate that cached
decoding changes attention work from repeatedly processing the full prefix to only
processing the new token. It should explain concepts; it should not become a custom
training framework.
