from aerollm.common.schemas import EvidenceSpan, Split
from aerollm.evaluation.generation import aggregate_generation, score_generation
from aerollm.evaluation.schemas import ExampleProvenance, GroundedQAExample
from aerollm.generation.backends import (
    BackendIdentity,
    GenerationResult,
    GenerationTrace,
)


def _example() -> GroundedQAExample:
    return GroundedQAExample(
        "dev-1", "What happened?", "report", "event", "family", Split.DEVELOPMENT,
        True, "The aircraft landed safely.", (),
        (EvidenceSpan("chunk-1", "landed safely"),),
        ExampleProvenance("author", (), ("report",), True),
    )


def _result(text: str) -> GenerationResult:
    trace = GenerationTrace(
        BackendIdentity("fake", "model", "revision"), "prompt-v1", 0, 32, 0.0,
        (), 10, 5, 25.0,
    )
    return GenerationResult(text, trace)


def test_generation_metrics_are_transparent_and_citation_aware() -> None:
    score = score_generation(
        _example(), "rag", _result("The aircraft landed safely [1]."),
        retrieved_chunk_ids=("chunk-1", "chunk-2"), context_chunk_ids=("chunk-1",),
    )

    assert score.exact_match == 0.0
    assert score.token_f1 > 0.8
    assert score.gold_evidence_coverage == 1.0
    assert score.citation_precision == 1.0
    assert score.citation_recall == 1.0


def test_generation_aggregate_keeps_variants_separate() -> None:
    base = score_generation(_example(), "base", _result("unknown"))
    prompted = score_generation(_example(), "prompted", _result("The aircraft landed safely."))

    aggregate = aggregate_generation([base, prompted])["variants"]

    assert aggregate["base"]["token_f1"] == 0.0
    assert aggregate["prompted"]["exact_match"] == 1.0


def test_unanswerable_refusal_is_not_penalized_for_missing_citation() -> None:
    example = GroundedQAExample(
        "dev-negative", "What color?", "report", "event", "family",
        Split.DEVELOPMENT, False, None, ("Refuse unsupported premise",), (),
        ExampleProvenance("author", (), ("report",), True),
    )

    score = score_generation(
        example, "rag", _result("The report does not specify that information."),
        retrieved_chunk_ids=("chunk-1",), context_chunk_ids=("chunk-1",),
    )

    assert score.refusal_correct == 1.0
    assert score.citation_precision is None
    assert score.citation_recall is None
