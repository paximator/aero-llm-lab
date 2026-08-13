"""Framework-neutral serving contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class BackendAnswer:
    text: str
    backend: str


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]: ...


@runtime_checkable
class AnswerBackend(Protocol):
    def answer(
        self, question: str, passages: tuple[RetrievedPassage, ...], *,
        request_id: str, max_new_tokens: int, temperature: float, prompt_version: str,
    ) -> BackendAnswer: ...


@runtime_checkable
class Postprocessor(Protocol):
    def process(self, answer: str, passages: tuple[RetrievedPassage, ...]) -> str: ...
