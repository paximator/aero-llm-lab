"""Generation backend contracts."""

from aerollm.generation.backends import (
    BackendIdentity,
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    GenerationTrace,
    ModelBackend,
)
from aerollm.generation.pipeline import (
    GroundedPredictionV1,
    build_grounded_prediction,
    select_retrieved_chunks,
)
from aerollm.generation.transformers_backend import TransformersBackend, fingerprint_model_config

__all__ = [
    "BackendIdentity",
    "FakeBackend",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTrace",
    "GroundedPredictionV1",
    "ModelBackend",
    "TransformersBackend",
    "build_grounded_prediction",
    "fingerprint_model_config",
    "select_retrieved_chunks",
]
