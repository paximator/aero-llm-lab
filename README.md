# AeroLLM Lab

AeroLLM Lab is an evaluation-first LLM engineering project for grounded aviation
analysis. It demonstrates the complete lifecycle of a domain LLM: data provenance,
retrieval, post-training, evaluation, inference, deployment, and benchmarking.

This is not a general-purpose aviation chatbot. The initial product surface is a
small set of evidence-backed tasks over public aviation safety reports: structured
fact extraction, cited question answering, and grounded incident summaries.

## Current results — development snapshot

- The data, retrieval, generation, QLoRA, evaluation, and typed FastAPI serving
  paths are implemented and covered by 233 tests on Windows/Linux CI.
- Hybrid retrieval reaches Recall@10 `0.700` on the locked historical test set;
  reranking reaches MRR@10 `0.483`.
- Pinned local Ministral 3 3B inference and NF4 QLoRA run within the measured RTX
  4070 Laptop 8 GB envelope.
- With identical reviewed gold context, QLoRA improves development accuracy from
  `18.5%` to `44.4%` and valid grounded output from `0%` to `37.0%`.
- End-to-end SFT+RAG remains a documented negative result: retrieval hit@3 is
  `63.0%`, but retrieved-context outputs do not yet pass strict grounding gates.
- Evaluation-suite v2 has 27 reviewed development examples; 45 test records still
  require independent human authoring and review before the suite can be frozen.
- Serving v1 provides fail-closed `/health`, `/ready`, and `/v1/answer` boundaries
  with deterministic CI fakes and lazy local-model initialization.

See the [development review brief](docs/development-review.md) for the claim
boundary, demo commands, limitations, results, and remaining work.

## Intended comparison

The target frozen comparison uses one task suite and prediction contract for:

1. minimal-instruct model;
2. prompted base model;
3. retrieval-augmented generation (RAG);
4. supervised fine-tuning (SFT with LoRA/QLoRA);
5. SFT plus RAG;
6. optional preference tuning only after the earlier baselines are stable.

Current local experiments pin `mistralai/Ministral-3-3B-Instruct-2512` for FP8
inference baselines and `mistralai/Ministral-3-3B-Base-2512` for NF4 QLoRA. Model
revisions and runtime configuration are recorded in tracked configs and result
reports. The five-system frozen comparison is not complete: its evaluation-v2 test
labels and SFT+RAG development gate remain outstanding.

## Design principles

- Evaluation precedes optimization.
- Raw source documents are immutable and traceable.
- Splits are performed by report or event, never by chunk.
- Generated examples cannot silently enter the test set.
- Answers requiring evidence expose verifiable citations.
- Experiments record data, prompt, model, adapter, index, and code versions.
- Added techniques must justify their complexity through measured improvement.

## Documentation

- [Development review brief](docs/development-review.md)
- [Review presentation and demo script](docs/development-review-presentation.md)
- [Operational action register](docs/action-register.md)
- [Repository architecture](docs/architecture.md)
- [Implementation roadmap](docs/roadmap.md)
- [Local hardware profile](docs/hardware-profile.md)
- [Data sources, NTSB account, and credential setup](docs/data-sources.md)
- [Project progress](docs/progress.md)
- [Experiment results](docs/results/README.md)
- [Retrieval annotation guide](docs/evaluation/retrieval-annotation-guide.md)

## Local setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked development environment:

```powershell
uv sync --locked --extra dev
uv run --locked aerollm-doctor
```

`uv.lock` is authoritative: do not install Torch or Triton manually with `pip`.
The core/dev profile works without a GPU. For NVIDIA retrieval and generation on
Windows or Linux, install and verify the separately pinned CUDA profile:

```powershell
uv sync --locked --extra dev --extra transformers
uv run --locked aerollm-doctor --transformers
```

The doctor prints every relevant version and an actionable error when the CUDA
wheel, driver, FP8 dtype, or Windows Triton runtime is wrong. Native Windows and
Linux NVIDIA systems are supported for Transformers experiments. The repository
does not claim vLLM support; it may be evaluated under WSL2/Linux later only if a
measured serving requirement justifies it. macOS remains suitable for core data,
evaluation, and unit-test work, not the CUDA benchmark path.

Create an account in the [NTSB Developer Portal](https://developer.ntsb.gov/) and
subscribe to the public API product to obtain a subscription key. Copy the tracked
template to the shared repository-level secrets directory. These commands derive
the correct location from Git and work from any worktree:

```powershell
$commonGitDir = git rev-parse --path-format=absolute --git-common-dir
$repositoryRoot = Split-Path -Parent $commonGitDir
New-Item -ItemType Directory -Force "$repositoryRoot\.secrets"
Copy-Item configs\secrets\ntsb.env.example "$repositoryRoot\.secrets\ntsb.env"
```

Edit `<repository-root>/.secrets/ntsb.env` and replace only the placeholder after
`AEROLLM_NTSB_API_KEY=`. Never put the key in a tracked TOML file, command argument,
issue, log, or chat.

Load it into the current PowerShell process without printing its value:

```powershell
. .\scripts\import-local-env.ps1 -Name ntsb
```

The loader automatically resolves the shared repository root, so the same secret
works from every Git/Cascade worktree. API configuration and endpoint paths remain
public in `configs/data/ntsb.toml`.

## Reproduction workflows

The sections below provide detailed reproduction entry points. The complete
development vertical slice now extends beyond data preparation through retrieval,
generation, QLoRA, evaluation, and serving; tracked result reports state which
gates passed or failed. Tests use recorded, reviewed, or synthetic inputs, and live
API calls remain explicit.

Create a reproducible chunk manifest from a parsed report with:

```powershell
uv run aerollm-chunk-document artifacts/parsed/ntsb/<prefix>/<digest>.json
```

Chunking parameters live in `configs/data/chunking.toml`. Manifests retain exact
source offsets, page provenance, content type, section metadata, and a configuration
fingerprint so retrieval experiments can be reproduced and audited.

Build a frozen pilot corpus from one or more report acquisition manifests with:

```powershell
uv run aerollm-build-corpus artifacts/manifests/ntsb-reports-*.json
```

The corpus builder verifies immutable artifact hashes, keeps complete NTSB events in
one deterministic train/development/test split, checks page coverage and chunk
quality, and writes a companion build manifest. Pilot settings are versioned in
`configs/data/corpus.toml`; generated corpora remain ignored local artifacts.

Plan a diverse, bounded set of reports before downloading PDFs:

```powershell
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19 --dry-run
```

The dry run makes bounded NTSB discovery requests and freezes the selected report
list in `artifacts/pilot/plan.json`; it does not download report PDFs. Review that
plan, then repeat the command without `--dry-run`. Completed reports are skipped on
retry, failures are isolated in `artifacts/pilot/failures.json`, and the final run
builds the frozen corpus. Use `--refresh-plan` only when intentionally replacing an
existing selection.

Materialize the reviewed frozen plan and build the corpus:

```powershell
uv run aerollm-build-pilot --start-date 2018-01-01 --end-date 2025-12-31 `
  --target-reports 19
```

Freeze the independently approved retrieval test packet and validate it against the
held-out corpus source:

```powershell
uv run aerollm-freeze-retrieval-test
uv run aerollm-validate-corpus artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_test_v1.json
```

The freeze command records `data/evaluation/retrieval_test_v1.sha256`, is idempotent
for identical content, and refuses to overwrite different bytes. Test questions,
answers, evidence, failures, and metrics are excluded from training and tuning.

Install the local Transformers stack with the platform-pinned CUDA build of Torch:

```powershell
uv sync --locked --extra transformers
```

Run the pinned Ministral 3 3B FP8 feasibility benchmark after downloading the
model snapshot to the ignored path shown below:

```powershell
uv run aerollm-generation-smoke artifacts/models/ministral-3-3b-instruct-2512 `
  --model-id mistralai/Ministral-3-3B-Instruct-2512 `
  --model-revision b35d4dfe56c142746f54dbd64f579faab2744308 `
  --fp8-kernel-revision 7cdb05d472d6c954c7d03182ed836ebfd4610df0 `
  --prompt "Explain why accident reports separate facts from analysis." `
  --max-new-tokens 64 `
  --output artifacts/benchmarks/generation/smoke.json
```

The first invocation includes Triton compilation. Run the identical command again
to measure a warm kernel cache, and keep both artifacts distinct.

Run the development-only base, prompted, and RAG comparison with the pinned local
artifacts. The command checkpoints after every answer and resumes identical runs:

```powershell
uv run aerollm-evaluate-generation `
  --config configs/evaluation/generation_dev_v1.toml `
  --dataset data/evaluation/retrieval_dev_v1.json `
  --corpus artifacts/corpora/ntsb-pilot-v1.json `
  --dense-index artifacts/indexes/e5-small-v2 `
  --dense-model artifacts/models/intfloat-e5-small-v2 `
  --reranker-model artifacts/models/cross-encoder-ms-marco-minilm-l6-v2 `
  --generation-model artifacts/models/ministral-3-3b-instruct-2512 `
  --output artifacts/evaluation/generation_dev_v1.json `
  --review-output artifacts/evaluation/generation_dev_v1.review.json
```

The test configuration is digest-locked and additionally requires the explicit
`--allow-frozen-test` flag. Do not run it while selecting prompts or retrieval
settings; the exact one-time command and results are documented in
`docs/results/generation-baselines-v1.md`.

The dense retrieval baseline uses the pinned `intfloat/e5-small-v2` revision in
`configs/retrieval/dense_e5_small_v2.toml`. Model weights, persistent embeddings,
and raw run reports remain under ignored `artifacts/` paths.

Generate the metadata-only corpus-v2 event review packet from verified cached NTSB
snapshots (add `--discover` after loading the ignored NTSB environment when more
metadata is required):

```powershell
uv run aerollm-select-corpus-v2 `
  --source-corpus artifacts/corpora/ntsb-pilot-v1.json
```

Review `artifacts/pilot/corpus_v2_selection.review.json` using
`docs/evaluation/corpus-v2-selection-review-guide.md`. This stage does not download
PDFs or draft evaluation questions.

After the reviewed selection has been frozen and corpus v2 materialized, generate
the 72-slot annotation workbook with:

```powershell
uv run aerollm-build-suite-v2-review
```

The 27 development questions have completed the current model-assisted review
stage. The 45 label-free test slots and blank public authoring template are tracked
under `data/evaluation/v2/`; filled test gold belongs only in the ignored private
workbook and must be independently human-authored and reviewed. Follow
`docs/evaluation/evaluation-suite-v2-annotation-review-guide.md` and the explicit
status boundary in `docs/evaluation/evaluation-suite-v2-status.md`.

Do not pass `--refresh-plan` during materialization: that flag intentionally
replaces the reviewed selection. Generated source snapshots, parsed documents,
chunks, failure reports, and corpora are stored under ignored `artifacts/` paths.
If an official report URL is unavailable, successful reports remain resumable and
the sanitized reason is recorded in `artifacts/pilot/failures.json`. After reviewing
those failures, build a validated corpus from the completed manifests with:

```powershell
$manifests = Get-ChildItem artifacts/manifests/pilot -Filter *.json |
  Sort-Object Name | ForEach-Object FullName
uv run aerollm-build-corpus @manifests
```
