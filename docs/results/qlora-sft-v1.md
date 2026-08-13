# QLoRA SFT v1

Status: pipeline, memory-smoke, tiny-overfit, and first 50-record run completed.

## Reproducible setup

- Base model: `mistralai/Ministral-3-3B-Base-2512`.
- Revision: `a58f2a5360e75b8249fd7f44508b72ac9d11fe89`.
- Quantization: bitsandbytes NF4, double quantization, BF16 compute.
- LoRA: rank 16, alpha 32, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Optimizer: paged AdamW 8-bit; learning rate `2e-4`; seed 42.
- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB nominal VRAM.
- Runtime: Python 3.13.7, Torch 2.7.1+cu128, CUDA 12.8.

The Base checkpoint has no chat template. The runner therefore uses an explicit,
version-controlled `[SYSTEM]`, `[USER]`, `[ASSISTANT]` rendering and masks the
system/user prefix from the causal-LM loss. Model loading occurs only inside the
runner. Adapters and full loss traces remain in ignored `artifacts/training/`.

## Gates and measurements

| Gate | Records | Steps | Initial loss | Final loss | Peak VRAM | Time | Reload |
|---|---:|---:|---:|---:|---:|---:|---|
| Memory smoke | 1 | 1 | 2.101889 | 2.101889 | 4,375.54 MiB | 9.99 s | Pass |
| Tiny overfit | 8 | 40 | 0.396694 | 0.000239 | 4,171.12 MiB | 41.74 s | Pass |
| First SFT run | 50 | 150 | 0.396694 | 0.000020 | 4,375.23 MiB | 179.18 s | Pass |

The first-run adapter digest is
`sha256:6ad104d2b666caa2ee32b4700b984ef089c571b978828f31372786df3b9a4bee`.

Run command:

```powershell
uv run --extra transformers --extra training aerollm-train-qlora `
  --model artifacts/models/Ministral-3-3B-Base-2512 `
  --output artifacts/training/qlora-sft-50-v1 `
  --records 50 --steps 150
```

## Interpretation and limitations

The results prove that the pinned Base checkpoint can be quantized to NF4, trained
within the local 8 GB envelope, saved, fingerprinted, and reloaded. The strong loss
decrease proves deliberate fitting of the evidence-linked records.

This is not yet a model-quality claim. There is no held-out SFT validation loss,
the 50 examples are narrow grounded-extraction records, and the run revisits each
record three times. The next required evidence is generation after adapter reload,
followed by a paired Base/SFT and RAG/SFT+RAG comparison on development examples.
