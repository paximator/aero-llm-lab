"""Framework-neutral serving contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float
    event_id: str | None = None
    report_id: str | None = None


@dataclass(frozen=True, slots=True)
class BackendAnswer:
    text: str
    backend: str


@dataclass(frozen=True, slots=True)
class ServingCitation:
    chunk_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class ProcessedServingAnswer:
    answer: str | None
    citations: tuple[ServingCitation, ...]
    abstained: bool
    abstention_reason: str | None = None


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        event_id: str | None = None,
        report_id: str | None = None,
    ) -> tuple[RetrievedPassage, ...]: ...


@runtime_checkable
class AnswerBackend(Protocol):
    def answer(
        self,
        question: str,
        passages: tuple[RetrievedPassage, ...],
        *,
        request_id: str,
        max_new_tokens: int,
        temperature: float,
        prompt_version: str,
    ) -> BackendAnswer: ...


@runtime_checkable
class Postprocessor(Protocol):
    def process(
        self, answer: str, passages: tuple[RetrievedPassage, ...]
    ) -> ProcessedServingAnswer: ...


@dataclass(frozen=True, slots=True)
class ServingDependencies:
    backend: AnswerBackend
    retriever: Retriever
    postprocessor: Postprocessor


DependencyInitializer = Callable[[], ServingDependencies]


class ScopeValidationError(ValueError):
    pass
