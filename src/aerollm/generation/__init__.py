"""Generation backend contracts."""

from aerollm.generation.backends import (
    BackendIdentity,
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    GenerationTrace,
    ModelBackend,
)

__all__ = [
    "BackendIdentity",
    "FakeBackend",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTrace",
    "ModelBackend",
]
