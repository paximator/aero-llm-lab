"""Ranking metrics and reports for the lexical retrieval baseline."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from aerollm.common.documents import Chunk
from aerollm.common.schemas import Split
from aerollm.evaluation.schemas import EvaluationDataset, GroundedQAExample
from aerollm.retrieval.bm25 import BM25Index
from aerollm.retrieval.schemas import RetrievalResult

_IDENTIFIER = re.compile(r"\b(?=[A-Z0-9-]{5,}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b")


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


@dataclass(frozen=True, slots=True)
class RetrievalExampleScore:
    example_id: str
    relevant_chunk_ids: tuple[str, ...]
    result: RetrievalResult
    recall_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "result": self.result.to_dict(),
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
        }


@dataclass(frozen=True, slots=True)
class RetrievalAggregate:
    example_count: int
    scored_count: int
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    mean_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "scored_count": self.scored_count,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_latency_ms": self.mean_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    dataset_id: str
    dataset_version: str
    index_manifest: dict[str, object]
    k: int
    scores: tuple[RetrievalExampleScore, ...]
    aggregate: RetrievalAggregate
    slices: dict[str, RetrievalAggregate]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "index_manifest": self.index_manifest,
            "k": self.k,
            "scores": [score.to_dict() for score in self.scores],
            "aggregate": self.aggregate.to_dict(),
            "slices": {key: value.to_dict() for key, value in sorted(self.slices.items())},
            "evaluator": "deterministic-retrieval-v1",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        rows = [
            "# Lexical retrieval report",
            "",
            f"- Dataset: `{self.dataset_id}` version `{self.dataset_version}`",
            f"- Index: `{self.index_manifest['index_id']}`",
            f"- Cutoff: `k={self.k}`",
            "",
            "| Slice | Examples | Scored | Recall@k | MRR | nDCG@k | Mean latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        groups = {"overall": self.aggregate, **self.slices}
        for name, score in groups.items():
            rows.append(
                f"| {name} | {score.example_count} | {score.scored_count} | "
                f"{_format_metric(score.recall_at_k)} | {_format_metric(score.mrr)} | "
                f"{_format_metric(score.ndcg_at_k)} | {score.mean_latency_ms:.3f} |"
            )
        return "\n".join(rows) + "\n"


def _score(example: GroundedQAExample, result: RetrievalResult) -> RetrievalExampleScore:
    relevant = tuple(dict.fromkeys(span.chunk_id for span in example.evidence))
    if not relevant:
        return RetrievalExampleScore(example.example_id, relevant, result, None, None, None)
    ranks = {hit.chunk_id: hit.rank for hit in result.hits}
    found_ranks = sorted(ranks[chunk_id] for chunk_id in relevant if chunk_id in ranks)
    recall = len(found_ranks) / len(relevant)
    reciprocal_rank = 1.0 / found_ranks[0] if found_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in found_ranks)
    ideal_count = min(len(relevant), len(result.hits))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return RetrievalExampleScore(
        example.example_id, relevant, result, recall, reciprocal_rank, dcg / ideal
    )


def _aggregate(scores: Sequence[RetrievalExampleScore]) -> RetrievalAggregate:
    scored = [score for score in scores if score.recall_at_k is not None]
    count = len(scores)
    if not count:
        raise ValueError("cannot aggregate an empty retrieval slice")
    return RetrievalAggregate(
        example_count=count,
        scored_count=len(scored),
        recall_at_k=sum(score.recall_at_k for score in scored) / len(scored) if scored else None,
        mrr=sum(score.reciprocal_rank for score in scored) / len(scored) if scored else None,
        ndcg_at_k=sum(score.ndcg_at_k for score in scored) / len(scored) if scored else None,
        mean_latency_ms=sum(score.result.latency_ms for score in scores) / count,
    )  # type: ignore[arg-type]


def evaluate_retrieval(
    dataset: EvaluationDataset,
    index: BM25Index,
    chunks: tuple[Chunk, ...],
    *,
    k: int = 10,
) -> RetrievalReport:
    """Search each non-training question and compute exact gold-chunk metrics."""

    if any(example.split is Split.TRAIN for example in dataset.examples):
        raise ValueError("retrieval evaluation accepts development or frozen test examples only")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    scores = tuple(
        _score(example, index.search(example.question, k=k)) for example in dataset.examples
    )
    groups: defaultdict[str, list[RetrievalExampleScore]] = defaultdict(list)
    for example, score in zip(dataset.examples, scores, strict=True):
        groups[f"answerable:{str(example.answerable).lower()}"].append(score)
        groups[f"split:{example.split.value}"].append(score)
        query_kind = "identifier" if _IDENTIFIER.search(example.question) else "natural-language"
        groups[f"query:{query_kind}"].append(score)
        kinds = {
            chunk_by_id[chunk_id].kind
            for chunk_id in score.relevant_chunk_ids
            if chunk_id in chunk_by_id
        }
        if kinds:
            for kind in kinds:
                groups[f"content:{kind.value}"].append(score)
        else:
            groups["content:none"].append(score)
    return RetrievalReport(
        dataset.dataset_id,
        dataset.version,
        index.manifest.to_dict(),
        k,
        scores,
        _aggregate(scores),
        {name: _aggregate(group) for name, group in groups.items()},
    )
