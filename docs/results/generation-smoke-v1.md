# Ministral 3 3B generation smoke v1

## Decision

The official Ministral 3 3B Instruct FP8 checkpoint runs fully on the locally
detected RTX 4070 Laptop GPU without CPU offload. It is suitable for correctness
experiments and small generation evaluations. At the observed 45 W GPU limit,
native-Windows Transformers inference is not yet an interactive-serving baseline;
serving throughput must be measured separately under WSL2/Linux and vLLM.

## Reproducible configuration

- Model: `mistralai/Ministral-3-3B-Instruct-2512`
- Model revision: `b35d4dfe56c142746f54dbd64f579faab2744308`
- FP8 kernel revision: `7cdb05d472d6c954c7d03182ed836ebfd4610df0`
- Configuration: `configs/generation/ministral_3_3b_instruct_2512.toml`
- Runtime: Python 3.13, Torch 2.7.1+cu128, CUDA 12.8, Triton Windows 3.7.1
- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB nominal VRAM,
  45 W power limit observed in the hardware profile
- Prompt tokens: 19; generated tokens: 64; greedy decoding with KV caching

The exact command is documented in the repository README. Raw JSON outputs live
under ignored `artifacts/benchmarks/generation/`.

## Measurements

| Run | Load | Generation | Throughput | Peak allocated VRAM | Reserved VRAM |
|---|---:|---:|---:|---:|---:|
| Cold Triton cache | 4.23 s | 99.45 s | 0.64 token/s | 4.96 GB | 4.98 GB |
| Warm Triton cache | 3.33 s | 57.09 s | 1.12 token/s | 4.96 GB | 4.98 GB |

GB values use decimal bytes as emitted by the benchmark. Generation time covers
prefill and decoding together; it is not time-to-first-token.

## Interpretation and limitations

- The feasibility gate passes with roughly 3.2 GB of nominal VRAM left for larger
  contexts and evaluation plumbing, subject to allocator and display overhead.
- Cold-start compilation is material and must not be reported as steady-state
  throughput.
- This is one prompt and one warm repetition, not a statistically stable serving
  benchmark. A later benchmark must add warm-up, repeated prompts, percentiles,
  time-to-first-token, input-token throughput, power/thermal capture, and vLLM.
- The detected GPU is an RTX 4070 Laptop GPU. All local capacity decisions use
  runtime-detected hardware.
