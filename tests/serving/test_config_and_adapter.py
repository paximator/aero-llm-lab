import sys
from pathlib import Path

import pytest

from aerollm.generation import BackendIdentity, FakeBackend
from aerollm.serving.adapters import ExistingMistralAdapter
from aerollm.serving.bootstrap import ProductionConfig
from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import RetrievedPassage


def test_repository_serving_config_loads() -> None:
    config = ServingConfig.from_toml(Path("configs/serving/v1.toml"))

    assert config.service_version == "v1"
    assert config.top_k == 4
    assert config.temperature == 0.0
    assert config.max_concurrency == 1


def test_existing_mistral_adapter_uses_preconstructed_backend() -> None:
    backend = FakeBackend(
        {"question": "answer"},
        identity=BackendIdentity("transformers-ministral-fp8", "mistral", "commit"),
    )
    adapter = ExistingMistralAdapter(backend)

    result = adapter.answer(
        "question", (RetrievedPassage("one", "context", 1.0),),
        request_id="req", max_new_tokens=12, temperature=0.0, prompt_version="serving-v1",
    )

    assert result.text == "answer"
    assert result.backend == "transformers-ministral-fp8:mistral"
    assert "aerollm.generation.ministral_backend" not in sys.modules
    assert "vllm" not in result.backend.casefold()


def test_serving_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ServingConfig(top_k=0)
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        ServingConfig(request_timeout_seconds=0.0)


def test_repository_production_config_loads_without_loading_model() -> None:
    config = ProductionConfig.from_toml(Path("configs/serving/production_v1.toml"))

    assert config.corpus_path == Path("artifacts/corpora/ntsb-pilot-v1.json")
    assert config.model_id == "mistralai/Ministral-3-3B-Instruct-2512"
    assert config.retrieval_mode == "hybrid_reranked"
    assert config.dense_model_id == "intfloat/e5-small-v2"
    assert config.reranker_candidate_k == 20
    assert config.fp8_kernel_path == Path("artifacts/kernels/finegrained-fp8")
    assert "aerollm.generation.ministral_backend" not in sys.modules
