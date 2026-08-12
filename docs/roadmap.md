# Implementation roadmap

The roadmap is organized around demonstrable increments. Each phase ends with an
artifact and an explicit gate; later techniques are added only when their baseline
and evaluation are trustworthy.

## Phase 0 — Frame the experiment

**Deliverables**

- Choose one or two initial task contracts and define valid outputs.
- Identify candidate public report sources and document usage constraints.
- Write evaluation rubrics, dataset schemas, and an experiment manifest schema.
- Select the initial open-weight Mistral checkpoint and hardware budget.
- Add Python packaging, linting, tests, and a reproducible command interface.

**Exit gate:** a reviewer can tell what success means, what data may be used, what
cannot enter the test set, and how a run is reproduced.

## Phase 1 — Data and evaluation foundation

**Deliverables**

- Acquire a small corpus with URLs, checksums, timestamps, and provenance.
- Parse text while retaining page/section positions; measure parser/OCR quality.
- Normalize, deduplicate, and create report-level train/dev/test splits.
- Hand-author and independently check a compact gold evaluation set.
- Implement deterministic metrics, failure categories, and a Markdown/HTML report.

**Exit gate:** the build is repeatable from a manifest, leakage checks pass, gold
examples point to verifiable evidence, and a trivial baseline produces a report.

## Phase 2 — Base, prompted, and retrieval baselines

**Deliverables**

- Run base and versioned prompted baselines through one inference interface.
- Implement lexical and dense retrieval, then a measured hybrid baseline.
- Evaluate chunking and Recall@k before generation quality.
- Add reranking only after documenting baseline retrieval failures.
- Run RAG with page/section citations and abstention behavior.

**Exit gate:** all variants use identical examples; retrieval and generation
failures are separable; claimed gains include paired results and uncertainty.

## Phase 3 — Parameter-efficient SFT

**Deliverables**

- Build evidence-linked SFT records with automated validation and manual samples.
- Establish a LoRA recipe; add QLoRA if the hardware budget requires it.
- Track configs, revisions, seeds, curves, wall time, and peak VRAM.
- Evaluate SFT closed-book and SFT+RAG on the frozen suite.
- Ablate data volume, adapter settings, and prompt format where useful.

**Exit gate:** results explain where SFT helps, hurts, or duplicates RAG; the
adapter and evaluation are reproducible; regressions are categorized.

## Phase 4 — Serving and systems characterization

**Deliverables**

- Expose typed FastAPI endpoints with streaming and trace IDs.
- Serve with vLLM under documented model and quantization settings.
- Add smoke/load tests and measure cold/warm latency, throughput, and VRAM.
- Compare quality versus latency under fixed load shapes.
- Containerize the serving path and document hardware assumptions.

**Exit gate:** a fresh environment launches the service and reproduces a published
benchmark within an agreed tolerance.

## Phase 5 — Bounded tools and agent evaluation

**Deliverables**

- Add typed, read-only tools with validation and timeouts.
- Implement bounded orchestration with full traces and iteration limits.
- Evaluate tool choice, arguments, recovery, and end-to-end task success.
- Compare against direct prompting/RAG on tasks that genuinely need tools.

**Exit gate:** tools produce a measured task-success improvement and failures are
observable and safely terminated; otherwise retain tools without an agent loop.

## Phase 6 — Educational Transformer track

This track can proceed independently once the main evaluation foundation is stable.

**Deliverables**

- Implement and explain RMSNorm, causal attention, RoPE, and transformer blocks.
- Add autoregressive generation and a preallocated KV cache.
- Test shapes, masking, cached/uncached equivalence, and cache growth.
- Profile decoding with and without caching.

**Exit gate:** tests establish mathematical and behavioral correctness and the
documentation connects the implementation to production inference.

## Phase 7 — Optional preference tuning

**Prerequisite:** SFT evaluation reveals a stable, preference-shaped failure mode,
and preference labels can be collected consistently.

**Deliverables**

- Define a preference rubric and audit annotator agreement.
- Build evidence-linked chosen/rejected pairs without test contamination.
- Run DPO against the same SFT checkpoint and frozen evaluation suite.
- Report gains, regressions, calibration, and training cost.

**Exit gate:** DPO supplies a statistically and practically meaningful benefit; if
not, publish the negative result and retain the simpler SFT system.

## Recommended first release

The strongest early portfolio release is Phases 0–3 plus a minimal Phase 4 serving
path. It demonstrates the coherent core: trustworthy data, leakage-aware evaluation,
retrieval diagnostics, parameter-efficient post-training, and controlled comparison
of base/prompted/RAG/SFT/SFT+RAG. Agents, DPO, and the educational track deepen that
story rather than delaying it.

## Cross-cutting definition of done

For every phase:

- commands run from configuration rather than edited notebooks;
- tests cover critical transformations and schemas;
- artifacts carry provenance and deterministic IDs;
- logs do not expose sensitive document contents or credentials;
- failure examples are retained for regression testing;
- documentation states hardware, model revision, data version, and limitations;
- generated reports include aggregate metrics and representative failures.
