"""Adapters for existing runtime boundaries; importing this module never loads a model."""

from __future__ import annotations

from aerollm.generation.backends import GenerationRequest, ModelBackend
from aerollm.serving.contracts import BackendAnswer, RetrievedPassage


class ExistingMistralAdapter:
    """Serve an already-constructed local generation backend.

    Construction and model loading stay with ``MinistralBackend.from_local_path``;
    this adapter deliberately accepts only an existing backend instance.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def answer(
        self, question: str, passages: tuple[RetrievedPassage, ...], *,
        request_id: str, max_new_tokens: int, temperature: float, prompt_version: str,
    ) -> BackendAnswer:
        result = self._backend.generate(GenerationRequest(
            prompt=question, context=tuple(passage.text for passage in passages),
            max_new_tokens=max_new_tokens, temperature=temperature,
            prompt_version=prompt_version, request_id=request_id,
        ))
        identity = result.trace.identity
        return BackendAnswer(result.text, f"{identity.backend}:{identity.model_id}")
