"""Pinned local CUDA backend for the official Ministral 3 FP8 checkpoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aerollm.generation.backends import (
    BackendIdentity,
    GenerationRequest,
    GenerationResult,
    GenerationTrace,
)


class MinistralBackend:
    def __init__(
        self, model: Any, tokenizer: Any, torch: Any, *, identity: BackendIdentity,
    ) -> None:
        self._model, self._tokenizer, self._torch = model, tokenizer, torch
        self._identity = identity

    @classmethod
    def from_local_path(
        cls, path: Path, *, model_id: str, model_revision: str, fp8_kernel_revision: str,
    ) -> MinistralBackend:
        import torch
        from kernels import get_kernel
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        from transformers.integrations import finegrained_fp8

        if not path.is_dir() or not torch.cuda.is_available():
            raise ValueError("an existing model directory and CUDA are required")
        tokenizer = MistralCommonBackend.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
        )
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype="auto", device_map="cuda",
        ).eval()
        kernel = get_kernel(
            "kernels-community/finegrained-fp8", revision=fp8_kernel_revision,
            trust_remote_code=True,
        )
        finegrained_fp8._FINEGRAINED_FP8 = finegrained_fp8.FineGrainedFP8(
            matmul=kernel.matmul_2d, batched_matmul=kernel.matmul_batched,
            grouped_matmul=kernel.matmul_grouped,
        )
        return cls(
            model, tokenizer, torch,
            identity=BackendIdentity("transformers-ministral-fp8", model_id, model_revision),
        )

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        content = "\n\n".join((*request.context, request.prompt))
        messages = [{"role": "user", "content": content}]
        encoded = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        input_ids = input_ids.to("cuda")
        self._torch.cuda.synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                input_ids, max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0, use_cache=True,
            )
        self._torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1_000
        completion = generated[0, input_ids.shape[-1]:]
        return GenerationResult(
            self._tokenizer.decode(completion, skip_special_tokens=True),
            GenerationTrace(
                self.identity, request.prompt_version, request.seed,
                request.max_new_tokens, request.temperature, tuple(request.context),
                int(input_ids.shape[-1]), int(completion.shape[-1]), latency_ms,
                request.request_id, request.metadata,
                {"torch": self._torch.__version__, "cuda": str(self._torch.version.cuda)},
            ),
        )
