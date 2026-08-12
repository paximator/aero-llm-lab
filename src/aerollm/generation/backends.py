"""Inference contracts shared by local model runtimes and evaluations.

This module intentionally has no dependency on an inference framework.  A later
Transformers or vLLM adapter only needs to implement :class:`ModelBackend` and
translate the immutable request into its runtime's inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable


def _immutable_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    """The exact inference implementation and artifacts used for a run."""

    backend: str
    model_id: str
    model_revision: str
    adapter_id: str | None = None
    adapter_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.backend.strip() or not self.model_id.strip() or not self.model_revision.strip():
            raise ValueError("backend, model_id, and model_revision are required")
        if self.adapter_revision is not None and self.adapter_id is None:
            raise ValueError("adapter_revision requires adapter_id")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    context: Sequence[str] = ()
    max_new_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    prompt_version: str = "unversioned"
    request_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version cannot be empty")
        if any(not passage.strip() for passage in self.context):
            raise ValueError("context passages cannot be empty")
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    """Replay-relevant provenance captured for every generated output."""

    identity: BackendIdentity
    prompt_version: str
    seed: int
    max_new_tokens: int
    temperature: float
    context: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    request_id: str | None = None
    request_metadata: Mapping[str, str] = field(default_factory=dict)
    backend_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "request_metadata", _immutable_mapping(self.request_metadata))
        object.__setattr__(self, "backend_metadata", _immutable_mapping(self.backend_metadata))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    trace: GenerationTrace

    @property
    def model_id(self) -> str:
        return self.trace.identity.model_id

    @property
    def prompt_tokens(self) -> int:
        return self.trace.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.trace.completion_tokens

    @property
    def latency_ms(self) -> float:
        return self.trace.latency_ms

    @property
    def metadata(self) -> Mapping[str, str]:
        """Compatibility alias for runtime-specific result metadata."""

        return self.trace.backend_metadata


@runtime_checkable
class ModelBackend(Protocol):
    """Synchronous local inference boundary.

    Serving layers may schedule/batch calls, while runtime adapters retain one
    comparable request/result contract.
    """

    @property
    def identity(self) -> BackendIdentity: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


FakeResponder = Callable[[GenerationRequest], str]


class FakeBackend:
    """Deterministic, dependency-free backend for evaluation and integration tests."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        responder: FakeResponder | None = None,
        identity: BackendIdentity | None = None,
    ) -> None:
        if responses is not None and responder is not None:
            raise ValueError("provide responses or responder, not both")
        self._responses = dict(responses or {})
        self._responder = responder
        self._identity = identity or BackendIdentity(
            backend="fake", model_id="deterministic-fake", model_revision="1"
        )

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        text = self._response_for(request)
        trace = GenerationTrace(
            identity=self.identity,
            prompt_version=request.prompt_version,
            seed=request.seed,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            context=tuple(request.context),
            prompt_tokens=_token_count(_render_input(request)),
            completion_tokens=_token_count(text),
            latency_ms=0.0,
            request_id=request.request_id,
            request_metadata=request.metadata,
            backend_metadata={"deterministic": "true"},
        )
        return GenerationResult(text=text, trace=trace)

    def _response_for(self, request: GenerationRequest) -> str:
        if self._responder is not None:
            return self._responder(request)
        if request.prompt in self._responses:
            return self._responses[request.prompt]

        payload = {
            "context": list(request.context),
            "max_new_tokens": request.max_new_tokens,
            "prompt": request.prompt,
            "seed": request.seed,
            "temperature": request.temperature,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"fake:{digest}"


def _render_input(request: GenerationRequest) -> str:
    return "\n".join((*request.context, request.prompt))


def _token_count(text: str) -> int:
    """Stable approximation for the fake; real adapters report tokenizer counts."""

    return len(text.split())
