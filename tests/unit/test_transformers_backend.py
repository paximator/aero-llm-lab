from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aerollm.generation import (
    BackendIdentity,
    GenerationRequest,
    ModelBackend,
    TransformersBackend,
    fingerprint_model_config,
)
from aerollm.generation import transformers_backend as module


class FakeIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class FakeEncoding(dict[str, Any]):
    moved_to: object | None = None

    def to(self, device: object) -> FakeEncoding:
        self.moved_to = device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.encoding = FakeEncoding(input_ids=FakeIds([10, 11, 12]))
        self.rendered: str | None = None

    def __call__(self, text: str, *, return_tensors: str) -> FakeEncoding:
        self.rendered = text
        assert return_tensors == "pt"
        return self.encoding

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "decoded:" + ",".join(str(value) for value in ids)


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **options: Any) -> list[list[int]]:
        self.options = options
        return [[10, 11, 12, 20, 21]]


def _backend(
    tmp_path: Path, **options: Any
) -> tuple[TransformersBackend, FakeModel, FakeTokenizer]:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    backend = TransformersBackend(
        model,
        tokenizer,
        identity=BackendIdentity("transformers-local", "tiny", "commit-abc"),
        model_path=tmp_path,
        config_fingerprint="sha256:config",
        **options,
    )
    return backend, model, tokenizer


def test_greedy_generation_maps_request_and_preserves_trace(tmp_path: Path) -> None:
    clock = iter((10.0, 10.025))
    backend, model, tokenizer = _backend(tmp_path, clock=lambda: next(clock))
    request = GenerationRequest(
        prompt="Question?",
        context=("Evidence.",),
        max_new_tokens=7,
        seed=4,
        prompt_version="qa-v1",
        request_id="example-1",
        metadata={"dataset_version": "v1"},
    )

    result = backend.generate(request)

    assert isinstance(backend, ModelBackend)
    assert tokenizer.rendered == "Evidence.\nQuestion?"
    assert tokenizer.encoding.moved_to == "cpu"
    assert model.options["max_new_tokens"] == 7
    assert model.options["do_sample"] is False
    assert "temperature" not in model.options
    assert result.text == "decoded:20,21"
    assert result.trace.prompt_tokens == 3
    assert result.trace.completion_tokens == 2
    assert result.trace.latency_ms == pytest.approx(25.0)
    assert result.trace.identity.model_revision == "commit-abc"
    assert result.trace.request_id == "example-1"
    assert result.trace.request_metadata == {"dataset_version": "v1"}
    assert result.trace.backend_metadata["config_fingerprint"] == "sha256:config"


def test_sampled_generation_uses_request_seed(tmp_path: Path) -> None:
    generator_calls: list[tuple[int, object]] = []

    def generator_factory(seed: int, device: object) -> str:
        generator_calls.append((seed, device))
        return "seeded-generator"

    backend, model, _ = _backend(
        tmp_path, clock=lambda: 1.0, generator_factory=generator_factory
    )
    backend.generate(GenerationRequest(prompt="Question?", temperature=0.7, seed=19))

    assert generator_calls == [(19, "cpu")]
    assert model.options["do_sample"] is True
    assert model.options["temperature"] == 0.7
    assert model.options["generator"] == "seeded-generator"


def test_config_fingerprint_is_stable_and_content_derived(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    first = fingerprint_model_config(tmp_path)
    assert first == fingerprint_model_config(tmp_path)

    (tmp_path / "tokenizer_config.json").write_text('{"add_bos_token":true}', encoding="utf-8")
    assert fingerprint_model_config(tmp_path) != first


def test_loader_is_local_only_and_disables_remote_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, Any]]] = []
    model = FakeModel()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **options: Any) -> FakeTokenizer:
            calls.append(("tokenizer", path, options))
            return FakeTokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **options: Any) -> FakeModel:
            calls.append(("model", path, options))
            return model

    fake_transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_transformers)

    backend = TransformersBackend.from_local_path(tmp_path, model_revision="local-revision")

    assert backend.identity.model_id == tmp_path.name
    assert model.eval_called
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2] == {"local_files_only": True, "trust_remote_code": False} for call in calls)


def test_loader_rejects_remote_or_incomplete_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing local directory"):
        TransformersBackend.from_local_path(
            "organization/remote-model", model_revision="revision"
        )
    with pytest.raises(ValueError, match="config.json"):
        fingerprint_model_config(tmp_path)
