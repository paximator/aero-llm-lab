# ADR 0004: Explicit local-only Transformers adapter

## Status

Accepted.

## Decision

Transformers support is optional and loaded lazily. `TransformersBackend` accepts
the shared generation request and produces the shared trace, including an explicit
model revision, tokenizer-derived token counts, decoding settings, seed, measured
latency, resolved local path, and SHA-256 configuration fingerprint.

The production constructor requires an existing directory containing `config.json`.
Tokenizer and causal model loading always sets `local_files_only=True` and
`trust_remote_code=False`. A repository name such as `organization/model` is not a
valid substitute for a local path, so adapter setup cannot implicitly download
weights or execute repository-provided Python code.

Greedy decoding is used when temperature is zero. Sampling uses a request-scoped
PyTorch generator seeded from the request, avoiding mutation of process-global RNG
state. Prompt construction remains deliberately basic until a versioned chat
template is selected and evaluated.

## Consequences

- Unit tests use runtime doubles and require no model, GPU, Transformers, or PyTorch.
- A smoke run is explicitly opt-in:
  `uv run --extra transformers aerollm-transformers-smoke PATH
  --model-revision REVISION --prompt "..."`.
- vLLM can later implement the same framework-neutral backend records after the
  local Transformers correctness baseline is measured.
