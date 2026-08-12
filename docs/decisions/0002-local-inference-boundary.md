# ADR 0002: Framework-neutral local inference boundary

## Status

Accepted.

## Decision

Local inference runtimes implement `ModelBackend.generate(GenerationRequest)` and
return a `GenerationResult` containing a `GenerationTrace`. The trace records the
backend, model and optional adapter revisions, prompt version, decoding settings,
seed, supplied context, request identity and metadata, token counts, and latency.

The core package does not import Transformers or vLLM. Future adapters translate
this contract at their package boundary and report tokenizer-derived counts and
measured latency. Model loading remains explicit adapter setup; importing or using
the contract never downloads weights.

`FakeBackend` provides scripted responses for fixtures and a canonical SHA-256
fallback for unscripted requests. It reports zero latency and a documented stable
whitespace token approximation, making repeated evaluation runs exactly comparable.

## Consequences

- Evaluation code can exercise the full generation boundary without GPUs, network
  access, nondeterministic timing, or model artifacts.
- Model and adapter revisions are mandatory provenance rather than optional log
  fields.
- Streaming and asynchronous batching stay outside this synchronous local boundary;
  serving adapters may schedule calls while preserving the same trace.
