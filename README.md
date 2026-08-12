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

## Current status

Phase 0: project design. The repository currently defines boundaries, artifacts,
and milestone gates. Implementation begins only after the initial task contract
and evaluation specification are fixed.
