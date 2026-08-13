"""Immutable contracts for deterministic generation postprocessing."""

from __future__ import annotations

from dataclasses import dataclass


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RawGeneration:
    """Unmodified text returned by a generation backend."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """One retrieved chunk available for grounding an answer."""

    chunk_id: str
    text: str

    def __post_init__(self) -> None:
        _non_empty(self.chunk_id, "chunk_id")
        _non_empty(self.text, "text")


@dataclass(frozen=True, slots=True)
class Citation:
    """A verbatim span attributed to a retrieved chunk."""

    chunk_id: str
    quote: str

    def __post_init__(self) -> None:
        _non_empty(self.chunk_id, "chunk_id")
        _non_empty(self.quote, "quote")


@dataclass(frozen=True, slots=True)
class ProcessedAnswer:
    """Validated answer plus the audit data produced while parsing it."""

    answer: str | None
    citations: tuple[Citation, ...]
    abstained: bool
    abstention_reason: str | None
    raw_output: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.abstained) is not bool:
            raise ValueError("abstained must be a boolean")
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
        if not isinstance(self.citations, tuple) or not all(
            isinstance(item, Citation) for item in self.citations
        ):
            raise ValueError("citations must be a tuple of Citation records")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item for item in self.warnings
        ):
            raise ValueError("warnings must be a tuple of non-empty strings")
        if self.abstained:
            if self.answer is not None or self.citations:
                raise ValueError("an abstention cannot contain an answer or citations")
            _non_empty(self.abstention_reason, "abstention_reason")
        else:
            _non_empty(self.answer, "answer")
            if not self.citations:
                raise ValueError("a grounded answer requires at least one citation")
            if self.abstention_reason is not None:
                raise ValueError("a grounded answer cannot have an abstention reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [
                {"chunk_id": citation.chunk_id, "quote": citation.quote}
                for citation in self.citations
            ],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "raw_output": self.raw_output,
            "warnings": list(self.warnings),
        }
