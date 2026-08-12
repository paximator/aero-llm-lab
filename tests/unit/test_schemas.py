from datetime import UTC, datetime

import pytest

from aerollm.common.schemas import EvidenceSpan, GroundedAnswer, SourceDocument


def test_source_document_requires_valid_digest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        SourceDocument(
            source="ntsb",
            source_id="case-1",
            source_url="https://example.test/case-1",
            publisher="NTSB",
            retrieved_at=datetime.now(UTC),
            sha256="invalid",
            artifact_path="snapshot.bin",
        )


def test_grounded_answer_requires_citation() -> None:
    with pytest.raises(ValueError, match="citation"):
        GroundedAnswer(answer="A factual answer", citations=(), abstained=False)


def test_abstention_cannot_claim_an_answer() -> None:
    with pytest.raises(ValueError, match="abstention"):
        GroundedAnswer(
            answer="Maybe",
            citations=(EvidenceSpan(chunk_id="chunk-1", quote="evidence"),),
            abstained=True,
            abstention_reason="insufficient evidence",
        )
