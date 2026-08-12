"""A single inference boundary for base, SFT, RAG, and hosted backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    context: Sequence[str] = ()
    max_new_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    metadata: Mapping[str, str] = field(default_factory=dict)


class ModelBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
