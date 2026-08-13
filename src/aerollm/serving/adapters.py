"""Adapters for existing runtime boundaries; importing this module never loads a model."""

from __future__ import annotations

from aerollm.generation.backends import GenerationRequest, ModelBackend
from aerollm.generation.grounded_prompt import (
    GROUNDED_RAG_PROMPT_VERSION,
    render_grounded_context,
    render_grounded_prompt,
)
from aerollm.postprocessing import PostprocessorV1, RetrievedEvidence
from aerollm.serving.contracts import (
    BackendAnswer,
    ProcessedServingAnswer,
    RetrievedPassage,
    ServingCitation,
)


class ExistingMistralAdapter:
    """Serve an already-constructed local generation backend.

    Construction and model loading stay with ``MinistralBackend.from_local_path``;
    this adapter deliberately accepts only an existing backend instance.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def answer(
        self,
        question: str,
        passages: tuple[RetrievedPassage, ...],
        *,
        request_id: str,
        max_new_tokens: int,
        temperature: float,
        prompt_version: str,
    ) -> BackendAnswer:
        if prompt_version != GROUNDED_RAG_PROMPT_VERSION:
            raise ValueError("serving requires the canonical grounded prompt version")
        result = self._backend.generate(
            GenerationRequest(
                prompt=render_grounded_prompt(question),
                context=render_grounded_context(
                    tuple((passage.chunk_id, passage.text) for passage in passages)
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_version=prompt_version,
                request_id=request_id,
            )
        )
        identity = result.trace.identity
        return BackendAnswer(result.text, f"{identity.backend}:{identity.model_id}")


class GroundedServingPostprocessor:
    """Map the canonical fail-closed postprocessor to the HTTP serving contract."""

    def __init__(self, processor: PostprocessorV1 | None = None) -> None:
        self._processor = processor or PostprocessorV1()

    def process(
        self, answer: str, passages: tuple[RetrievedPassage, ...]
    ) -> ProcessedServingAnswer:
        evidence = tuple(RetrievedEvidence(item.chunk_id, item.text) for item in passages)
        processed = self._processor.process(answer, evidence)
        return ProcessedServingAnswer(
            processed.answer,
            tuple(ServingCitation(item.chunk_id, item.quote) for item in processed.citations),
            processed.abstained,
            processed.abstention_reason,
        )
