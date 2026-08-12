# AeroLLM Lab

AeroLLM Lab is an evaluation-first LLM engineering project for grounded aviation
analysis. It is intended to demonstrate the complete lifecycle of a domain LLM:
data provenance, retrieval, post-training, evaluation, inference, deployment, and
systems benchmarking.

This is not a general-purpose aviation chatbot. The initial product surface is a
small set of evidence-backed tasks over public aviation safety reports, such as
extracting structured facts, answering questions with citations, and producing
grounded incident summaries.

## Intended comparisons

Every approach is evaluated against the same frozen task suite:

1. base model;
2. prompted base model;
3. retrieval-augmented generation (RAG);
4. supervised fine-tuning (SFT with LoRA/QLoRA);
5. SFT plus RAG;
6. optional preference tuning after the earlier baselines are stable.

The exact open-weight Mistral checkpoint is deliberately a configuration choice.
It will be selected during the baseline milestone based on license, context length,
tool-use support, hardware constraints, and reproducible availability.

## Design principles

- Evaluation precedes optimization.
- Raw source documents are immutable and traceable.
- Splits are performed by report or event, never by chunk.
- Generated examples cannot silently enter the test set.
- Answers that require evidence must expose citations.
- Experiments record data, prompt, model, adapter, index, and code versions.
- Each added technique must justify its complexity through measured improvement.

## Documentation

- [Repository architecture](docs/architecture.md)
- [Implementation roadmap](docs/roadmap.md)

## Current status

Phase 0: project design. The repository currently defines boundaries, artifacts,
and milestone gates; implementation follows only after the first task contract and
evaluation specification are fixed.
