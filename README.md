# AeroLLM Lab

AeroLLM Lab is an evaluation-first LLM engineering project for grounded aviation
analysis. It demonstrates the complete lifecycle of a domain LLM: data provenance,
retrieval, post-training, evaluation, inference, deployment, and benchmarking.

This is not a general-purpose aviation chatbot. The initial product surface is a
small set of evidence-backed tasks over public aviation safety reports: structured
fact extraction, cited question answering, and grounded incident summaries.

## Intended comparison

Every approach is evaluated against the same frozen task suite:

1. base model;
2. prompted base model;
3. retrieval-augmented generation (RAG);
4. supervised fine-tuning (SFT with LoRA/QLoRA);
5. SFT plus RAG;
6. optional preference tuning after the earlier baselines are stable.

The exact open-weight Mistral checkpoint remains a configuration choice until the
baseline milestone, where it will be selected based on license, context length,
tool-use support, hardware constraints, and reproducible availability.

## Design principles

- Evaluation precedes optimization.
- Raw source documents are immutable and traceable.
- Splits are performed by report or event, never by chunk.
- Generated examples cannot silently enter the test set.
- Answers requiring evidence expose verifiable citations.
- Experiments record data, prompt, model, adapter, index, and code versions.
- Added techniques must justify their complexity through measured improvement.

## Documentation

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
uv sync --extra dev
```

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

## Current status

The first end-to-end data slice is operational: NTSB API discovery, aviation case
metadata, formal report download, PDF parsing, and deterministic page-aware
chunking. Tests use recorded or synthetic inputs; live API calls remain explicit.

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
uv sync --extra transformers
```

The dense retrieval baseline uses the pinned `intfloat/e5-small-v2` revision in
`configs/retrieval/dense_e5_small_v2.toml`. Model weights, persistent embeddings,
and raw run reports remain under ignored `artifacts/` paths.

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
