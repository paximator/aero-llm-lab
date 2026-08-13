"""Grounded generation integration and unified prediction records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aerollm.common.documents import Chunk
from aerollm.generation.backends import GenerationResult, GenerationTrace
from aerollm.postprocessing import PostprocessorV1, ProcessedAnswer, RetrievedEvidence
from aerollm.retrieval.schemas import RetrievalResult


@dataclass(frozen=True, slots=True)
class GroundedPredictionV1:
    """One auditable result spanning retrieval, generation, and postprocessing."""

    query: str
    retrieval: RetrievalResult
    selected_chunk_ids: tuple[str, ...]
    output: ProcessedAnswer
    generation_trace: GenerationTrace

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.retrieval, RetrievalResult):
            raise ValueError("retrieval must be a RetrievalResult")
        if self.query != self.retrieval.query:
            raise ValueError("query must match the retrieval query")
        if not isinstance(self.selected_chunk_ids, tuple):
            raise ValueError("selected_chunk_ids must be a tuple")
        if len(set(self.selected_chunk_ids)) != len(self.selected_chunk_ids):
            raise ValueError("selected chunk IDs must be unique")
        retrieved_ids = tuple(hit.chunk_id for hit in self.retrieval.hits)
        positions = {chunk_id: index for index, chunk_id in enumerate(retrieved_ids)}
        if any(chunk_id not in positions for chunk_id in self.selected_chunk_ids):
            raise ValueError("selected chunks must come from retrieval hits")
        if tuple(sorted(self.selected_chunk_ids, key=positions.__getitem__)) != (
            self.selected_chunk_ids
        ):
            raise ValueError("selected chunks must preserve retrieval order")
        if not isinstance(self.output, ProcessedAnswer):
            raise ValueError("output must be a ProcessedAnswer")
        if not isinstance(self.generation_trace, GenerationTrace):
            raise ValueError("generation_trace must be a GenerationTrace")

    def to_dict(self) -> dict[str, object]:
        trace = self.generation_trace
        identity = trace.identity
        return {
            "schema_version": 1,
            "query": self.query,
            "retrieval": self.retrieval.to_dict(),
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "output": self.output.to_dict(),
            "generation_trace": {
                "identity": {
                    "backend": identity.backend,
                    "model_id": identity.model_id,
                    "model_revision": identity.model_revision,
                    "adapter_id": identity.adapter_id,
                    "adapter_revision": identity.adapter_revision,
                },
                "prompt_version": trace.prompt_version,
                "seed": trace.seed,
                "max_new_tokens": trace.max_new_tokens,
                "temperature": trace.temperature,
                "context": list(trace.context),
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "latency_ms": trace.latency_ms,
                "request_id": trace.request_id,
                "request_metadata": dict(trace.request_metadata),
                "backend_metadata": dict(trace.backend_metadata),
            },
        }


def build_grounded_prediction(
    generation: GenerationResult,
    retrieval: RetrievalResult,
    selected_chunks: Sequence[Chunk],
    *,
    postprocessor: PostprocessorV1 | None = None,
) -> GroundedPredictionV1:
    """Postprocess a generation using exactly the selected retrieved chunks."""

    if not isinstance(generation, GenerationResult):
        raise TypeError("generation must be a GenerationResult")
    if not isinstance(retrieval, RetrievalResult):
        raise TypeError("retrieval must be a RetrievalResult")
    chunks = tuple(selected_chunks)
    if not all(isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("selected_chunks must contain Chunk records")
    selected_ids = tuple(chunk.chunk_id for chunk in chunks)
    _validate_selection(selected_ids, retrieval)
    if generation.trace.context != tuple(chunk.text for chunk in chunks):
        raise ValueError("generation context must match the selected chunk texts")
    processor = postprocessor or PostprocessorV1()
    if not isinstance(processor, PostprocessorV1):
        raise TypeError("postprocessor must be a PostprocessorV1")
    evidence = tuple(RetrievedEvidence(chunk.chunk_id, chunk.text) for chunk in chunks)
    output = processor.process(generation.text, evidence)
    return GroundedPredictionV1(
        query=retrieval.query,
        retrieval=retrieval,
        selected_chunk_ids=selected_ids,
        output=output,
        generation_trace=generation.trace,
    )


def select_retrieved_chunks(
    retrieval: RetrievalResult, chunks: Mapping[str, Chunk]
) -> tuple[Chunk, ...]:
    """Resolve all retrieval hits to chunks in their deterministic rank order."""

    if not isinstance(retrieval, RetrievalResult):
        raise TypeError("retrieval must be a RetrievalResult")
    if not isinstance(chunks, Mapping):
        raise TypeError("chunks must be a mapping")
    resolved: list[Chunk] = []
    for hit in retrieval.hits:
        try:
            chunk = chunks[hit.chunk_id]
        except KeyError as exc:
            raise ValueError(f"retrieved chunk is unavailable: {hit.chunk_id}") from exc
        if not isinstance(chunk, Chunk) or chunk.chunk_id != hit.chunk_id:
            raise ValueError(f"invalid chunk mapping for retrieval hit: {hit.chunk_id}")
        resolved.append(chunk)
    return tuple(resolved)


def _validate_selection(selected_ids: tuple[str, ...], retrieval: RetrievalResult) -> None:
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected chunks must be unique")
    retrieved_ids = tuple(hit.chunk_id for hit in retrieval.hits)
    positions = {chunk_id: index for index, chunk_id in enumerate(retrieved_ids)}
    if any(chunk_id not in positions for chunk_id in selected_ids):
        raise ValueError("selected chunks must come from retrieval hits")
    if tuple(sorted(selected_ids, key=positions.__getitem__)) != selected_ids:
        raise ValueError("selected chunks must preserve retrieval order")
