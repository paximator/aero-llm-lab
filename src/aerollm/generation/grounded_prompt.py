"""Versioned prompt and context rendering shared by offline and serving RAG."""

from __future__ import annotations

from collections.abc import Sequence

GROUNDED_RAG_PROMPT_VERSION = "grounded-json-v1"
_INSTRUCTIONS = """Use only the supplied report excerpts. Return exactly one JSON object:
{
  "answer": "string or null",
  "citations": [{"chunk_id": "string", "quote": "exact span"}],
  "abstained": false,
  "abstention_reason": null
}
Every citation quote must be an exact span from its chunk. If evidence is insufficient,
set answer to null, citations to [], abstained to true, and give a short reason.

Question: """


def render_grounded_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return _INSTRUCTIONS + question.strip()


def render_grounded_context(passages: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    rendered = []
    for chunk_id, passage in passages:
        if not chunk_id.strip() or not passage.strip():
            raise ValueError("grounded context requires non-empty chunk IDs and text")
        rendered.append(f'<chunk id="{chunk_id}">\n{passage}\n</chunk>')
    return tuple(rendered)
