"""Evaluation dataset records and validation."""

from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry
from aerollm.evaluation.runner import EvaluationReport, PredictionRecord, evaluate
from aerollm.evaluation.schemas import (
    EvaluationDataset,
    ExampleProvenance,
    GroundedQAExample,
)

__all__ = [
    "EvaluationDataset",
    "EvaluationReport",
    "CorpusManifest",
    "ExampleProvenance",
    "GroundedQAExample",
    "PredictionRecord",
    "SourceManifestEntry",
    "evaluate",
]
