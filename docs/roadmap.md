# Implementation roadmap

This roadmap incorporates the January 2026 portfolio review. The repository has
enough framework infrastructure; priority now goes to a credible end-to-end
post-training comparison rather than additional horizontal features.

## Portfolio objective

The first portfolio release must let a reviewer trace one controlled experiment:

```text
minimal instruct → task-prompted → RAG → QLoRA SFT → SFT + RAG
```

Every system uses the same task contracts, prediction schema, evaluation runner,
and frozen test suite. Retrieval, generation, and post-training failures remain
separable. New capabilities cannot bypass this comparison.

## Priority rules

1. Use development data for every design or parameter choice.
2. Never tune from the existing or future frozen-test outputs.
3. Prefer data/evaluation quality over new framework layers or model count.
4. Add a technique only to address an observed failure category.
5. Preserve simple baselines and publish negative results.
6. Complete the post-training vertical slice before agents, DPO, or serving polish.

## P0 — First portfolio release

### P0.1 — Evaluation suite v2

The ten-example test set remains a valid pipeline baseline, but it is too small and
event-concentrated for strong model-quality claims. Build a new suite from events
that have not been used for prompt/retrieval selection.

**Deliverables**

- Approximately 50–100 manually reviewed examples.
- Several independent event/report families in each applicable split.
- Documented task/category counts: factual extraction, numeric/date, entity/ID,
  causal, multi-evidence, answerable, unanswerable, and citation-sensitive.
- Immutable evidence spans, report-level splitting, leakage validation, and digest.
- A migration note: v1 results remain historical and are not silently overwritten.

**Gate**

- Every test example has independent human review.
- No event family crosses train/development/test.
- All evidence resolves to canonical chunks.
- Test labels have not informed prompts, retrieval settings, or training data.

### P0.2 — Task-specific evaluation v2

Replace universal answer overlap as the headline measure while retaining it as a
diagnostic. Extend the schema with a task type and an explicit scoring strategy.

**Deliverables**

- Deterministic number/date/ID/categorical/yes-no/structured-field scorers.
- Key-fact or rubric scoring for open grounded QA.
- Citation validity, support, abstention, and answer correctness reported separately.
- Versioned semantic/manual evaluation protocol; no single opaque judge metric.
- Stable failure taxonomy:
  `retrieval_miss`, `retrieval_low_rank`, `context_truncation`,
  `answer_mismatch`, `unsupported_answer`, `missing_citation`,
  `incorrect_citation`, `missed_abstention`, `unexpected_abstention`,
  `format_violation`, and `generation_truncation`.

**Gate**

- Scoring dispatches by declared task type.
- Representative examples and edge cases are human-reviewed.
- The evaluator consumes the same prediction record for every system variant.

### P0.3 — Freeze the generation interface

The v1 base/prompted/RAG run is complete. Before training, close the remaining
provenance gaps without adding another inference framework.

**Deliverables**

- Rename the current “base” precisely as `minimal-instruct`; a true pretrained
  checkpoint is a separate optional baseline, not an alias.
- Immutable prompt records with ID, version, SHA-256, system prompt, answer schema,
  citation format, and abstention rule.
- Prediction records containing query, retrieval/reranking scores, selected chunk
  IDs, selected-context token count, prompt/completion tokens, citations,
  abstention, and runtime trace.
- A small experiment manifest containing git commit, command, dataset/model/prompt/
  retriever digests, hardware, prediction path, and metrics path.

**Gate**

- Minimal-instruct, prompted, and RAG replay through one evaluator.
- Retrieval misses and answer-generation failures render separately.
- No configuration change is selected from frozen-test outcomes.

### P0.4 — Evidence-linked SFT dataset

This is the highest-value missing engineering capability. Build training records
only from train-split reports; development examples may validate behavior but test
events are categorically excluded.

**Deliverables**

- Chat-formatted records with system/user/assistant messages.
- Report, event, source-chunk, generator, synthetic status, and review provenance.
- Validators for source/evidence existence, event split, duplicate content, schema,
  citations, and the configured token budget.
- Dataset digest, category distribution, length statistics, rejected-record report,
  and a manually reviewed sample.
- Staged volumes: approximately 50 records for pipeline/tiny-overfit validation,
  then about 250 high-quality records for the first meaningful run. Expand only if
  an ablation justifies it.

**Gate**

- Zero test-event contamination.
- Every retained factual answer is evidence-linked.
- A reviewer can trace any training record back to immutable report chunks.

### P0.5 — QLoRA training gates

Use the documented local hardware envelope and `uv`. WSL2/Linux is preferred if
bitsandbytes or trainer support is materially more reliable than native Windows.

**Gate A: memory smoke**

- Load the pinned Mistral-family checkpoint in 4-bit.
- Inject LoRA, run forward/backward and one optimizer step.
- Record GPU/driver/Torch/CUDA, RAM, sequence length, batch sizes, and peak VRAM.

**Gate B: tiny overfit**

- Deliberately overfit 8–32 validated records.
- Demonstrate a strong loss decrease and expected training behavior.
- Save, reload, fingerprint, and evaluate the adapter through the shared backend.

**Gate C: first real SFT run**

- Train on the first reviewed dataset using a pinned config and seed.
- Record dataset/model revisions, LoRA targets/rank/alpha, quantization, optimizer,
  learning rate/scheduler, effective batch size, steps, curves, wall time, peak
  VRAM, validation metrics, checkpoints, and adapter digest.

### P0.6 — Frozen comparison

Evaluate these systems on evaluation-suite v2 without changing its labels or the
selected configurations:

```text
minimal-instruct
prompted
RAG
SFT
SFT + RAG
```

**Analysis requirements**

- Identify formatting, terminology, extraction, abstention, and instruction gains.
- Separate report-specific grounding gains attributable to RAG.
- Detect SFT hallucination or memorized-prior regressions.
- Explain whether SFT+RAG improves use of retrieved evidence.
- Include representative failures and paired per-example comparisons.

**P0 exit gate**

- A reloadable adapter and its digest exist.
- All five systems use one evaluator and frozen suite.
- Headline results link to manifests, configs, and reports.
- Limitations and negative results are explicit.

## P1 — Credibility and presentation

Complete after the first SFT comparison, except for components already required by
P0 evaluation quality.

1. Add paired bootstrap confidence intervals and win/loss/tie counts.
2. Render per-task metrics and stable failure-taxonomy counts.
3. Add a README comparison table linked to detailed reports.
4. Harden experiment manifests and deterministic report rendering.
5. Run targeted data-volume or adapter ablations only when they answer a concrete
   question from the first SFT result.

## P2 — Systems and secondary demonstrations

1. The minimal non-streaming FastAPI v1 endpoint is complete with typed requests,
   request IDs, safe errors, injected boundaries, and deterministic fakes. Streaming
   remains deferred until an inference benchmark justifies it.
2. Do not claim vLLM support. Evaluate it under WSL2/Linux only if later serving
   requirements justify the dependency, then benchmark TTFT, tokens/s, p50/p95,
   requests/s, and peak VRAM.
3. Educational Transformer correctness and profiling: RMSNorm, RoPE, causal/GQA
   attention, autoregressive decoding, KV cache, cached/uncached equivalence, and
   measured decoding improvement.
4. Read-only tool calling only for tasks that direct RAG cannot solve; measure tool
   selection, arguments, unnecessary calls, recovery, and end-to-end success.

Avoid Kubernetes, elaborate orchestration, another vector database, or additional
retrieval techniques without measured justification.

## P3 — Optional preference tuning

Attempt DPO only if SFT exposes a stable preference-shaped failure such as
unsupported answer versus abstention, valid versus plausible citations, concise
versus verbose answers, or schema-valid versus invalid output. Compare SFT and
SFT+DPO on the same frozen suite. A documented negative result is acceptable.

## Immediate execution queue

Status legend: `NEXT` is immediately actionable, `BLOCKED` has an unmet dependency,
`GATED` must satisfy a measured criterion, and `DEFERRED` is deliberately outside
the current milestone.

| ID | Status | Priority | Action | Exit criterion |
|---|---|---:|---|---|
| A1 | `NEXT` | P0 | Human-author and independently review the 45 prepared evaluation-v2 test slots. | Every slot has complete gold fields and a reviewer distinct from its author. |
| A2 | `BLOCKED` by A1 | P0 | Run schema, evidence, leakage, duplicate, target, and coverage gates; freeze the suite. | All gates pass and final bytes plus SHA-256 are recorded. |
| A3 | `NEXT` in parallel | P0 | Author reviewed train-only retrieved-context records for exact citation copying and balanced abstention. | A reviewed dataset has traceable chunks, no test families, and no automatic-label claim. |
| A4 | `GATED` by A3 | P0 | Train the next micro-adapter and rerun the five-example SFT+RAG development gate. | At least 4/5 outputs are contract-valid and grounded; otherwise stop. |
| A5 | `BLOCKED` by A2 and A4 | P0 | Run the frozen five-system comparison, paired failures, and uncertainty analysis. | One immutable suite and evaluator cover all five systems without test-driven tuning. |
| A6 | `DEFERRED` | P1/P2 | Streaming, vLLM evaluation, DPO, agents, and production serving benchmarks. | Re-prioritized only from a measured requirement after P0. |

Execution details, expected owners, evidence, and hand-off criteria live in the
[action register](action-register.md). This table is the canonical priority order;
the longer P0–P3 sections explain intent and acceptance gates.

The evaluation expansion is deliberately bounded: it strengthens claims but must
not become another framework-building phase that postpones post-training.

## Definition of done for the next major milestone

- [ ] Evaluation-suite v2 covers multiple independent report families.
- [ ] Approximately 50–100 examples are manually reviewed and frozen.
- [ ] Task-specific deterministic and rubric scoring is versioned.
- [ ] One prediction schema supports all five system variants.
- [ ] SFT records are evidence-linked, validated, and test-leakage-free.
- [x] QLoRA memory smoke passes.
- [x] Tiny-overfit, adapter save, reload, and development evaluation pass.
- [x] The first real SFT adapter and manifest exist.
- [ ] SFT and SFT+RAG run on the same frozen suite as the baselines.
- [ ] Results include uncertainty, failure categories, examples, and limitations.
- [ ] README headline metrics link to reproducible reports.
