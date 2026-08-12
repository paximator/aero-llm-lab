"""Generation backend contracts."""

from aerollm.generation.backends import (
    BackendIdentity,
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    GenerationTrace,
    ModelBackend,
)
from aerollm.generation.transformers_backend import TransformersBackend, fingerprint_model_config

__all__ = [
    "BackendIdentity",
    "FakeBackend",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTrace",
    "ModelBackend",
    "TransformersBackend",
    "fingerprint_model_config",
]
