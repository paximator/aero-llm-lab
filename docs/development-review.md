# Development review brief

Status at commit: development vertical slice implemented; evaluation-suite v2 and
end-to-end SFT+RAG quality gates remain incomplete.

## What is demonstrably working

- Snapshot-first NTSB ingestion, PDF parsing, deterministic chunking, corpus
  validation, and event-family split controls are implemented and tested.
- Retrieval has locked baselines. On the historical 10-example test set, BM25
  Recall@10 is 0.500, hybrid Recall@10 is 0.700, and reranked MRR@10 is 0.483.
- Local Ministral inference works on the RTX 4070 Laptop GPU. The recorded smoke
  result used 4.96 GB peak VRAM and generated 1.12 token/s after warm-up.
- QLoRA works inside the 8 GB envelope: memory smoke, tiny overfit, adapter
  save/reload, deterministic shuffled training, and artifact hashing all pass.
- With identical reviewed gold context on 27 development questions, QLoRA v1 raised
  task accuracy from 18.5% to 44.4% and valid grounded output from 0% to 37.0%.
- Serving v1 exposes typed `/health`, `/ready`, and `/v1/answer` endpoints with
  request IDs, latency, bounded concurrency, timeouts, safe errors, injected
  protocols, deterministic CI fakes, and lazy production initialization.

## Negative results retained on purpose

- A 90-record corrective SFT run regressed development accuracy to 33.3%; it was
  rejected and QLoRA v1 retained.
- In the real retrieved-context comparison, gold evidence hit@3 was 63.0%, but both
  RAG and SFT+RAG failed closed on all 27 development examples.
- Failure analysis separated Base non-JSON output from SFT generation truncation.
  Two prompt/context mini-fixes produced 0/5 valid outputs and were rejected.
- A 12-record EOS-aware retrieved-context micro-SFT removed truncation but still
  produced 0/5 valid grounded outputs because of exact-citation, schema, and
  unexpected-abstention failures. The gate stopped the larger run.

These outcomes show working experiment controls and honest model-selection gates;
they do not support a production-quality RAG claim.

## Evaluation status and claim boundary

Evaluation-suite v2 has 27 reviewed development examples. Its 45 test slots have a
published ready-to-fill structure, but still require independent human authoring
and review. The correct suite status remains
`human_review_and_test_authoring_required`. Development results may guide debugging;
they are not frozen-test model-quality evidence.

The strongest current model result is therefore the gold-context QLoRA v1
development comparison. The complete SFT+RAG path is implemented and auditable,
but its quality gate is explicitly negative.

## Five-minute reviewer path

Install and run the CPU checks:

```powershell
uv sync --locked --extra dev --extra serving
uv run --locked aerollm-doctor
uv run --locked ruff check .
uv run --locked python -m pytest
```

Start the deterministic serving demo; this loads no model:

```powershell
uv run --locked --extra serving uvicorn aerollm.serving.app:create_app `
  --factory --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/answer -Method Post `
  -ContentType application/json -Body '{"question":"What happened?"}'
```

The real local adapter boundary is opt-in and lazy:

```powershell
$env:AEROLLM_SERVING_CONFIG = "configs/serving/production_v1.toml"
uv run --locked --extra serving --extra transformers uvicorn `
  aerollm.serving.bootstrap:create_production_app --factory
```

It requires the ignored local corpus, indexes, model, and reranker paths configured
in `production_v1.toml`. The repository does not claim vLLM integration.

## Review entry points

- Current delivery narrative: [`progress.md`](progress.md)
- Priorities and exit gates: [`roadmap.md`](roadmap.md)
- Versioned results: [`results/README.md`](results/README.md)
- Evaluation-v2 scope: [`evaluation/evaluation-suite-v2-status.md`](evaluation/evaluation-suite-v2-status.md)
- Serving implementation: `src/aerollm/serving/`
- Serving tests: `tests/serving/`
- Operational work queue: [`action-register.md`](action-register.md)
- Presentation script and demo checklist:
  [`development-review-presentation.md`](development-review-presentation.md)

## Remaining priorities

1. Independently author and review the 45 evaluation-v2 test records, then freeze
   the suite and its digest.
2. Author reviewed retrieved-context training examples that teach exact citation
   copying and balanced abstention; do not scale automatic recomposition.
3. Re-run the five-system comparison only after its development mini-gate passes.
4. Add uncertainty and paired failure analysis once a frozen test suite exists.
5. Run and record a production serving benchmark only after model-quality and local
   artifact prerequisites are satisfied.

Deferred by design: agents, DPO, vLLM, distributed serving, and production-scale
observability. None is needed to review the current engineering and evaluation
discipline.
