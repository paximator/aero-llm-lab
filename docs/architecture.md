# Repository architecture

## Scope

The first vertical slice supports one defensible workflow end to end:

> Given a public aviation safety report, answer a factual question or produce a
> constrained summary, cite the supporting passages, and abstain when the report
> does not support the answer.

This narrow scope exercises ingestion, parsing, retrieval, generation, grounding,
evaluation, and serving. Taxonomy classification, causal-factor extraction,
cross-report synthesis, and tool-assisted analysis can follow once this slice is
reliable. Safety reports are historical evidence, not operational flight guidance;
the API and demo must state that boundary.

## Implemented v0.1 flow

```text
NTSB snapshot -> verified PDF -> page-aware Document -> stable Chunk
                                                        |
                    event-family split -> frozen Corpus -> retrieval/index
                                                        |             |
                                     frozen Examples -> generation -> metrics
```

Every boundary carries source, report, and event identity plus content hashes.
Corpus v2 uses version `ntsb-corpus-v2`; its reviewed metadata selection freezes
whole event families before question authoring. Its suite workbook contains 27
development drafts awaiting review and 45 deliberately unauthored test slots.

The following layout is the longer-term target, not a claim that every subsystem
exists in version 0.1.

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
│   └── decisions/            # architecture decision records
├── src/aerollm/
│   ├── data/                 # acquire, parse, normalize, deduplicate, split
│   ├── retrieval/            # chunk, index, retrieve, rerank
│   ├── generation/           # prompts and inference adapters
│   ├── training/             # dataset formatting and PEFT entry points
│   ├── evaluation/           # task runners, metrics, slices, reports
│   ├── agents/               # typed tools and bounded orchestration
│   ├── serving/              # FastAPI schemas and vLLM integration
│   ├── benchmarking/         # latency, throughput, quality/cost measurement
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
│   ├── datasets/             # versioned, human-reviewed evaluation records
│   ├── rubrics/              # explicit grading and abstention rules
│   └── baselines/            # checked-in aggregate results, not large outputs
├── scripts/                  # thin entry points; logic stays in src/
├── notebooks/               # exploration only, never canonical pipelines
├── reports/                  # generated experiment summaries and plots
└── infra/                    # containers and deployment manifests when needed
```

Large reports, weights, embeddings, indexes, and run outputs do not belong in Git.
Git stores manifests, hashes, small fixtures, configurations, and summarized
results; an artifact store or ignored local directory holds large objects.

## System boundaries

```text
public reports
     │
     ▼
acquire → parse/OCR → normalize → deduplicate → document-level split
     │                                      │
     │                                      ├── frozen evaluation examples
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
experiment. The evaluation protocol must explicitly state whether it measures
corpus search or closed-corpus QA with the target report supplied.

## Core contracts

Subsystems exchange stable typed records rather than ad hoc dictionaries:

- `SourceDocument`: URL, publisher, date, usage note, retrieval timestamp,
  checksum, report/event ID, and local artifact reference.
- `Document`: canonical text, page/section spans, and parser metadata.
- `Chunk`: stable ID, document ID, text, structural location, and offsets.
- `Example`: task, input, answer/rubric, evidence spans, split, author, provenance.
- `RetrievalResult`: query, ranked chunks, scores, index version, and timing.
- `GenerationTrace`: model/adapter IDs, prompt version, decoding parameters,
  retrieved context, tool calls, output, token counts, timings, and seed.
- `EvaluationResult`: example ID, system variant, metrics, failure labels,
  evaluator versions, and trace reference.

Derived artifacts use content-derived or deterministic identifiers. An experiment
manifest binds Git commit, configuration, source manifest, dataset and split
versions, model revision, adapter, index, and environment.

## Evaluation design

Evaluation is a first-class package, not the final notebook. The initial suite is a
small, manually verified gold set:

| Task | Primary measures | Important slices |
|---|---|---|
| Structured extraction | field F1, exact match | missing fields, tables, OCR |
| Report-grounded QA | correctness, citation precision/recall | answerable/unanswerable |
| Constrained summary | factual coverage, unsupported-claim rate | long reports, multiple causes |
| Retrieval | Recall@k, MRR/nDCG, latency | lexical, semantic, identifier queries |
| Tool use (later) | selection, argument validity, task success | failures, ambiguous input |

LLM-as-judge scores are calibrated against a human-reviewed subset and reported
separately from deterministic metrics. Paired outputs support bootstrap confidence
intervals and per-example regressions rather than mean scores alone.

Leakage controls are mandatory: split by event/report family, deduplicate before
splitting, track source dates, exclude synthetic data from evaluation, and inspect
near-duplicate overlap. Since pretraining may include public reports, results must
distinguish adaptation from memorization and favor supplied-evidence tasks.

## Retrieval, generation, and training

Start with transparent lexical and dense retrieval, then hybrid fusion. Add a
reranker only after retrieval error analysis shows the need. Preserve page and
section metadata through chunking so citations remain verifiable.

All principal variants implement one inference interface, keeping prompts,
decoding, traces, and evaluation comparable. RAG and fine-tuning are orthogonal
configuration choices rather than separate applications.

SFT data is produced by a reviewable pipeline and retains evidence links. Begin
with LoRA and use QLoRA when memory measurements justify it. Record trainable
parameter count, token distribution, packing, optimizer, precision, hardware,
wall time, peak VRAM, and seeds. DPO is gated on a clear rubric and an observed,
preference-shaped SFT failure mode.

## Agent, serving, and benchmarking boundaries

Agentic behavior solves bounded analysis tasks, not open-ended chat. Candidate
typed, read-only tools include report lookup, passage search, metadata filtering,
unit conversion, and statistics. Validate arguments, cap iterations, expose traces,
simulate failures, and compare task success with a non-agentic baseline.

FastAPI owns transport, validation, health/readiness, and trace IDs; vLLM owns
batched execution. An engine abstraction keeps local tests small or mocked.

Agent, FastAPI, and vLLM boundaries are planned; they are not implemented version
0.1 outcomes.

Benchmarks report time to first token, inter-token latency, end-to-end percentiles,
tokens and requests per second, peak allocated/reserved VRAM, prompt/output lengths,
concurrency, batching, quantization, hardware, and model revision. Cold and warm
runs are separate, with the exact load shape published.

## Educational Transformer

`transformer_lab/` remains independent of the production stack. It implements
causal multi-head attention, RoPE, RMSNorm, residual/MLP blocks, autoregressive
generation, and a preallocated KV cache. Tests compare cached and uncached logits,
verify masking and RoPE properties, and demonstrate the decoding work saved by the
cache. It explains production concepts; it is not a custom training framework.

This is a design boundary. The educational Transformer is not complete in version
0.1.
