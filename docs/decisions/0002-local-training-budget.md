# ADR 0002: Optimize the complete local track for 8 GB VRAM

- Status: accepted
- Date: 2026-08-12

## Context

Runtime inspection reports an RTX 4070 Laptop GPU with 8,188 MiB VRAM and a 45 W
power limit. The machine owner reports 64 GB DDR5 system RAM. This differs from the
initial expectation of a laptop RTX 4090 and must govern reproducible planning.

## Decision

Ministral 3 3B is the primary checkpoint family for the complete local comparison.
SFT uses QLoRA by default, with LoRA retained as an ablation only if it fits. Initial
training uses 1,024-token sequences and micro-batches of one. Larger checkpoints
are gated scale experiments and cannot block the core portfolio result.

The project prefers WSL2/Linux for CUDA training and vLLM, managed with uv. Every
run captures observed hardware rather than trusting this static planning profile.

## Consequences

- Base, prompt, RAG, SFT, and SFT+RAG remain locally demonstrable on one Mistral
  family.
- The experiment emphasizes rigor and reproducibility over headline parameter size.
- Long-context and larger-model experiments require later memory measurements or a
  remote GPU.
- CPU offload is allowed but must be labeled because it changes systems results.
