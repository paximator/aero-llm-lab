# Implementation roadmap

The roadmap is organized around demonstrable increments. Each phase ends with an
artifact and an explicit gate; later techniques are added only when their baseline
and evaluation are trustworthy.

## Phase 0 — Frame the experiment

**Deliverables**

- Choose one or two initial task contracts and define valid outputs.
- Identify candidate public report sources and document usage constraints.
- Write evaluation rubrics, dataset schemas, and an experiment manifest schema.
- Select the initial open-weight Mistral checkpoint and target hardware budget.
- Add Python packaging, linting, tests, and a reproducible command interface.

**Exit gate:** a reviewer can tell exactly what success means, what data may be
used, what cannot enter the test set, and how a run will be reproduced.

## Phase 1 — Data and evaluation foundation

**Deliverables**

- Acquire a small source corpus with URLs, checksums, timestamps, and provenance.
- Parse text while preserving page/section positions; measure parser/OCR quality.
- Normalize, deduplicate, and create report-level train/dev/test splits.
- Hand-author and double-check a compact gold evaluation set.
- Implement deterministic metrics, failure categories, and an HTML/Markdown report.

**Exit gate:** the dataset build is repeatable from a manifest; leakage checks pass;
gold examples point to verifiable evidence; a trivial baseline produces a report.

## Phase 2 — Base, prompted, and retrieval baselines

**Deliverables**

- Run the base model and a versioned prompted baseline through one interface.
- Implement lexical and dense retrieval, then a measured hybrid baseline.
- Evaluate chunking choices and retrieval Recall@k before generation quality.
- Add reranking only after documenting baseline retrieval failures.
- Run the RAG system with page/section citations and abstention behavior.

**Exit gate:** base, prompted, and RAG results are comparable on identical examples;
retrieval and generation failures can be separated; every claimed gain includes
paired results and uncertainty.

## Phase 3 — SFT with parameter-efficient tuning

**Deliverables**

- Build evidence-linked SFT records with automated validation and manual samples.
- Establish a LoRA recipe; add QLoRA if required by the hardware budget.
- Track configs, dataset/model revisions, seeds, loss curves, time, and peak VRAM.
- Evaluate SFT closed-book and SFT+RAG using the frozen suite.
- Perform ablations for data volume, adapter settings, and prompt format where useful.

**Exit gate:** the project can explain where SFT helps, hurts, or duplicates RAG;
the adapter and its evaluation are reproducible; regressions are categorized.

## Phase 4 — Serving and systems characterization

**Deliverables**

- Package inference behind a typed FastAPI API with streaming and trace IDs.
- Serve through vLLM with documented model and quantization settings.
- Add smoke/load tests and measure cold/warm latency, throughput, and VRAM.
- Produce quality-versus-latency comparisons under fixed load shapes.
- Containerize the reproducible serving path and document hardware assumptions.

**Exit gate:** a fresh environment can launch the service and reproduce a published
benchmark within an agreed tolerance.

## Phase 5 — Bounded tools and agent evaluation

**Deliverables**

- Add a small registry of typed, read-only tools with validation and timeouts.
- Implement a bounded orchestration loop with full traces and iteration limits.
- Create tool-choice, argument, recovery, and end-to-end task evaluations.
- Compare the agent against direct prompting/RAG on tasks that genuinely need tools.

**Exit gate:** tools create a measured task-success improvement, and failure cases
are observable and safely terminated. Otherwise retain tools without an agent loop.

## Phase 6 — Educational Transformer track

This can proceed independently once the main evaluation foundation is stable.

**Deliverables**

- Implement and explain RMSNorm, causal attention, RoPE, and transformer blocks.
- Add autoregressive generation and a preallocated KV cache.
- Test tensor shapes, masking, cached/uncached equivalence, and cache growth.
- Provide a small profiling demonstration of decoding with and without caching.

**Exit gate:** tests establish mathematical and behavioral correctness, and the
documentation connects the implementation to the production inference stack.

## Phase 7 — Optional preference tuning

**Prerequisite:** SFT evaluation reveals a stable, preference-shaped failure mode,
and preference labels can be collected consistently.

**Deliverables**

- Define a preference rubric and audit annotator agreement.
- Build evidence-linked chosen/rejected pairs without test contamination.
- Run DPO against the same SFT checkpoint and frozen evaluation suite.
- Report quality gains, regressions, calibration, and training cost.

**Exit gate:** DPO provides a statistically and practically meaningful benefit. If
not, publish the negative result and keep the simpler SFT system.

## Recommended first release

The strongest early portfolio release is Phases 0–3 plus a minimal Phase 4 serving
path. It demonstrates the difficult, coherent core: trustworthy data, leakage-aware
evaluation, retrieval diagnostics, parameter-efficient post-training, and controlled
comparison of base/prompted/RAG/SFT/SFT+RAG. Agentic behavior, DPO, and the educational
track should deepen that story rather than delay it.

## Cross-cutting definition of done

For every phase:

- commands run from configuration rather than edited notebooks;
- tests cover critical transformations and schemas;
- artifacts carry provenance and deterministic IDs;
- logs contain no document or credential surprises;
- failure examples are retained for regression testing;
- documentation states hardware, model revision, data version, and limitations;
- generated reports include both aggregate metrics and representative failures.
