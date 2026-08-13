"""Deterministic serving dependencies for CI and local contract checks."""

import hashlib

from aerollm.serving.contracts import BackendAnswer, RetrievedPassage


class FakeRetriever:
    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        return tuple(
            RetrievedPassage(
                f"fake-{digest}-{index}", f"Evidence {index} for: {query}", 1.0 / index,
            )
            for index in range(1, min(top_k, 2) + 1)
        )


class FakeAnswerBackend:
    def answer(
        self, question: str, passages: tuple[RetrievedPassage, ...], *,
        request_id: str, max_new_tokens: int, temperature: float, prompt_version: str,
    ) -> BackendAnswer:
        del request_id, max_new_tokens, temperature, prompt_version
        payload = "|".join((question, *(passage.chunk_id for passage in passages)))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return BackendAnswer(f"fake:{digest}", "deterministic-fake")


class IdentityPostprocessor:
    def process(self, answer: str, passages: tuple[RetrievedPassage, ...]) -> str:
        del passages
        return answer
