"""Opt-in adapter for causal language models stored on the local filesystem."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from aerollm.generation.backends import (
    BackendIdentity,
    GenerationRequest,
    GenerationResult,
    GenerationTrace,
)

_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class TransformersBackend:
    """A synchronous local-only Transformers causal-LM adapter."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        identity: BackendIdentity,
        model_path: Path,
        config_fingerprint: str,
        clock: Callable[[], float] = perf_counter,
        generator_factory: Callable[[int, Any], Any] | None = None,
    ) -> None:
        if identity.backend != "transformers-local":
            raise ValueError("TransformersBackend identity backend must be transformers-local")
        self._model = model
        self._tokenizer = tokenizer
        self._identity = identity
        self._model_path = model_path
        self._config_fingerprint = config_fingerprint
        self._clock = clock
        self._generator_factory = generator_factory or _torch_generator

    @classmethod
    def from_local_path(
        cls,
        model_path: str | Path,
        *,
        model_revision: str,
        model_id: str | None = None,
    ) -> TransformersBackend:
        path = _validated_model_path(model_path)
        if not model_revision.strip():
            raise ValueError("model_revision is required")
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "Transformers support is not installed; run "
                "`uv sync --extra transformers`"
            ) from error

        load_options = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), **load_options)
        model = transformers.AutoModelForCausalLM.from_pretrained(str(path), **load_options)
        model.eval()
        return cls(
            model,
            tokenizer,
            identity=BackendIdentity(
                backend="transformers-local",
                model_id=model_id or path.name,
                model_revision=model_revision,
            ),
            model_path=path,
            config_fingerprint=fingerprint_model_config(path),
        )

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        rendered = _render_input(request)
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None and hasattr(encoded, "to"):
            encoded = encoded.to(device)
        inputs = dict(encoded)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("tokenizer output did not contain input_ids")
        prompt_tokens = int(input_ids.shape[-1])
        options: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            options["temperature"] = request.temperature
            options["generator"] = self._generator_factory(request.seed, device)

        started = self._clock()
        generated = self._model.generate(**inputs, **options)
        latency_ms = (self._clock() - started) * 1_000
        sequence = generated[0]
        completion_ids = sequence[prompt_tokens:]
        completion_tokens = len(completion_ids)
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            trace=GenerationTrace(
                identity=self.identity,
                prompt_version=request.prompt_version,
                seed=request.seed,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                context=tuple(request.context),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                request_id=request.request_id,
                request_metadata=request.metadata,
                backend_metadata={
                    "config_fingerprint": self._config_fingerprint,
                    "model_path": str(self._model_path),
                },
            ),
        )


def fingerprint_model_config(model_path: str | Path) -> str:
    path = _validated_model_path(model_path)
    if not (path / "config.json").is_file():
        raise ValueError("local model directory must contain config.json")
    digest = hashlib.sha256()
    for name in _CONFIG_FILES:
        candidate = path / name
        if candidate.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(candidate.read_bytes())
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _validated_model_path(model_path: str | Path) -> Path:
    candidate = Path(model_path).expanduser()
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("model_path must be an existing local directory") from error
    if not path.is_dir():
        raise ValueError("model_path must be an existing local directory")
    return path


def _render_input(request: GenerationRequest) -> str:
    return "\n".join((*request.context, request.prompt))


def _torch_generator(seed: int, device: Any) -> Any:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "sampled generation requires the transformers dependency group"
        ) from error
    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    return generator.manual_seed(seed)


def _trace_to_dict(result: GenerationResult) -> Mapping[str, Any]:
    trace = result.trace
    return {
        "text": result.text,
        "trace": {
            "backend": trace.identity.backend,
            "model_id": trace.identity.model_id,
            "model_revision": trace.identity.model_revision,
            "prompt_version": trace.prompt_version,
            "seed": trace.seed,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "latency_ms": trace.latency_ms,
            "backend_metadata": dict(trace.backend_metadata),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a local Transformers model")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    backend = TransformersBackend.from_local_path(
        arguments.model_path, model_revision=arguments.model_revision
    )
    result = backend.generate(
        GenerationRequest(
            prompt=arguments.prompt,
            max_new_tokens=arguments.max_new_tokens,
            seed=arguments.seed,
            prompt_version="manual-smoke-v1",
        )
    )
    print(json.dumps(_trace_to_dict(result), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as an opt-in smoke command
    raise SystemExit(main())
