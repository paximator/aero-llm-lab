# Local hardware profile

## Development target

The local ML path is designed around the hardware visible on 2026-08-12:

| Resource | Budget |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Dedicated VRAM | 8,188 MiB |
| GPU power limit observed by `nvidia-smi` | 45 W |
| System RAM | 64 GB DDR5 (owner-reported) |
| Host | Windows 11 |

The GPU identity, VRAM, driver, free disk, and available memory must be captured
again in every benchmark manifest. The repository does not assume that the model
can consume all nominal VRAM: display, driver, allocator, and kernel overhead need
headroom.

## Model policy

All generative experiment variants use open-weight Mistral-family checkpoints. The
primary complete local track uses the Ministral 3 3B family:

- base and prompted baselines;
- RAG generation;
- QLoRA SFT and SFT+RAG;
- bounded tool-use experiments;
- optional preference tuning only after an isolated feasibility measurement.

An exact model revision is pinned in the experiment manifest before weights are
downloaded. Larger Mistral/Ministral 7B–8B checkpoints are scale experiments for
quantized inference and, if measurements permit, constrained QLoRA. Models at 14B
and above are remote-GPU experiments, not local completion requirements.

Retrieval remains task-appropriate: BM25 is the mandatory model-free baseline and
dense retrieval may use a dedicated embedding model. “Mistral-only” applies to the
generative LLM comparison; it does not justify using a causal generator as an
inferior embedding model.

## Local training envelope

The first QLoRA feasibility run starts conservatively:

- 4-bit NF4 weights with double quantization;
- BF16 compute when the runtime capability check passes, otherwise FP16;
- sequence length 1,024;
- micro-batch size 1 with gradient accumulation;
- gradient checkpointing;
- LoRA rank 16 and alpha 32 as a starting point;
- paged 8-bit optimizer where supported;
- evaluation and checkpoint cadence chosen to avoid memory spikes.

Sequence length, LoRA targets, rank, and effective batch size are measured variables,
not promises. The first run is a memory smoke test, followed by a tiny overfit test,
then a short end-to-end training run. Peak allocated/reserved VRAM, wall time,
tokens/second, thermals where available, and failures are recorded.

## Runtime environment

Linux under WSL2 is the preferred training and vLLM environment because the core
CUDA LLM ecosystem targets Linux. Windows remains the host and repository-control
environment. Python environments and commands use uv in both contexts.

On native Windows and Linux, the `transformers` dependency group resolves Torch
2.7.1 from the CUDA 12.8 PyTorch index. Native Windows additionally pins
`triton-windows` 3.7.1 because the official Ministral 3 FP8 checkpoint requires
UE8M0 support and the v4 fine-grained FP8 kernel. This platform mapping is pinned
in `pyproject.toml` and `uv.lock`; verify it with
`torch.cuda.is_available()` before recording GPU measurements.

CPU offload may use the 64 GB system RAM to make constrained experiments possible,
but offloaded results are reported separately: system RAM increases capacity, not
GPU bandwidth, and can materially reduce throughput.

## Explicit non-goals

- Full-parameter fine-tuning of multi-billion-parameter models locally
- Pretending an offloaded 14B model is a representative local serving benchmark
- Comparing local and remote runs without recording their different hardware
- Selecting a model solely because it barely loads while leaving no training margin
