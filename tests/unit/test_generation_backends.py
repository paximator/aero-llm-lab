import pytest

from aerollm.generation import (
    BackendIdentity,
    FakeBackend,
    GenerationRequest,
    ModelBackend,
)


def test_fake_backend_is_deterministic_and_satisfies_boundary() -> None:
    backend = FakeBackend()
    request = GenerationRequest(
        prompt="What happened?",
        context=("The aircraft landed safely.",),
        seed=42,
        prompt_version="grounded-qa-v1",
        request_id="example-7",
        metadata={"dataset_version": "gold-v1"},
    )

    first = backend.generate(request)
    second = backend.generate(request)

    assert isinstance(backend, ModelBackend)
    assert first == second
    assert first.text.startswith("fake:")
    assert first.trace.identity == backend.identity
    assert first.trace.prompt_version == "grounded-qa-v1"
    assert first.trace.seed == 42
    assert first.trace.context == ("The aircraft landed safely.",)
    assert first.trace.request_id == "example-7"
    assert first.trace.request_metadata == {"dataset_version": "gold-v1"}
    assert first.trace.backend_metadata == {"deterministic": "true"}
    assert first.latency_ms == 0.0


def test_fake_backend_changes_fallback_for_replay_relevant_inputs() -> None:
    backend = FakeBackend()
    baseline = backend.generate(GenerationRequest(prompt="question", seed=1)).text

    assert backend.generate(GenerationRequest(prompt="question", seed=2)).text != baseline
    with_context = backend.generate(
        GenerationRequest(prompt="question", context=("evidence",))
    ).text
    assert with_context != baseline


def test_fake_backend_supports_scripted_evaluation_outputs() -> None:
    backend = FakeBackend({"known question": '{"answer":"known"}'})

    result = backend.generate(GenerationRequest(prompt="known question"))

    assert result.text == '{"answer":"known"}'
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 1


def test_request_copies_mutable_traceability_inputs() -> None:
    context = ["evidence"]
    metadata = {"example_id": "one"}
    request = GenerationRequest(prompt="question", context=context, metadata=metadata)
    context.append("later")
    metadata["example_id"] = "changed"

    assert request.context == ("evidence",)
    assert request.metadata == {"example_id": "one"}
    with pytest.raises(TypeError):
        request.metadata["example_id"] = "mutation"  # type: ignore[index]


def test_identity_rejects_untraceable_or_incomplete_adapter() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        BackendIdentity(backend="transformers", model_id="model", model_revision="")
    with pytest.raises(ValueError, match="adapter_revision"):
        BackendIdentity(
            backend="transformers",
            model_id="model",
            model_revision="commit",
            adapter_revision="adapter-commit",
        )
