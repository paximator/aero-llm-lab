"""Run development-only base, prompted, and RAG generation evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tomllib
from pathlib import Path

from aerollm.common.schemas import Split
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.generation import (
    GenerationExampleResult,
    aggregate_generation,
    dumps,
    review_packet,
    score_generation,
)
from aerollm.evaluation.schemas import EvaluationDataset
from aerollm.generation.backends import GenerationRequest
from aerollm.retrieval.bm25 import BM25Index
from aerollm.retrieval.dense import DenseIndex
from aerollm.retrieval.dense_transformers import TransformersDenseEncoder
from aerollm.retrieval.hybrid import HybridIndex
from aerollm.retrieval.rerank import RerankedIndex
from aerollm.retrieval.rerank_transformers import TransformersCrossEncoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dense-index", type=Path, required=True)
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--generation-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-frozen-test", action="store_true")
    args = parser.parse_args(argv)

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    dataset_bytes = args.dataset.read_bytes()
    dataset = EvaluationDataset.from_json(dataset_bytes.decode("utf-8"))
    split = _require_allowed_split(
        dataset, config, dataset_bytes=dataset_bytes,
        allow_frozen_test=args.allow_frozen_test,
    )
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode("utf-8"))
    corpus.validate_dataset(dataset)
    examples = dataset.examples[:args.limit] if args.limit else dataset.examples
    retrievals = _retrieve(examples, corpus, corpus_bytes, args, config)

    identity = {
        "experiment": config["version"],
        "split": split.value,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model": config["model"],
    }
    scores = _resume_scores(args.output, identity)
    completed = {(score.example_id, score.variant) for score in scores}
    expected = len(examples) * len(config["variants"])
    if len(completed) == expected:
        _write_outputs(args.output, args.review_output, identity, scores)
        return 0

    from aerollm.generation.ministral_backend import MinistralBackend

    model_config = config["model"]
    backend = MinistralBackend.from_local_path(
        args.generation_model, model_id=model_config["id"],
        model_revision=model_config["revision"],
        fp8_kernel_revision=model_config["fp8_kernel_revision"],
    )
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    for example in examples:
        hits = retrievals[example.example_id]
        context_ids = tuple(hits[:int(config["rag_context_chunks"])])
        for variant in config["variants"]:
            if (example.example_id, variant) in completed:
                continue
            context = ()
            if variant == "rag":
                context = tuple(
                    f"[{rank}] chunk_id={chunk_id} pages={chunks[chunk_id].page_start}-"
                    f"{chunks[chunk_id].page_end}\n{chunks[chunk_id].text}"
                    for rank, chunk_id in enumerate(context_ids, start=1)
                )
            prompt = config["prompts"][variant].format(
                question=example.question, event_id=example.event_id,
                report_id=example.report_id,
            )
            result = backend.generate(
                GenerationRequest(
                    prompt, context, int(config["max_new_tokens"]),
                    float(config["temperature"]), int(config["seed"]),
                    f"{config['version']}:{variant}", example.example_id,
                    {"split": split.value, "variant": variant},
                )
            )
            scores.append(
                score_generation(
                    example, variant, result,
                    retrieved_chunk_ids=tuple(hits) if variant == "rag" else (),
                    context_chunk_ids=context_ids if variant == "rag" else (),
                )
            )
            print(f"completed {example.example_id} / {variant}", flush=True)
            _write_outputs(args.output, args.review_output, identity, scores)

    _write_outputs(args.output, args.review_output, identity, scores)
    return 0


def _require_allowed_split(
    dataset: EvaluationDataset, config: dict, *, dataset_bytes: bytes,
    allow_frozen_test: bool,
) -> Split:
    splits = {example.split for example in dataset.examples}
    if len(splits) != 1 or Split.TRAIN in splits:
        raise ValueError("generation evaluation requires exactly one non-training split")
    split = next(iter(splits))
    if config.get("allowed_split") != split.value:
        raise ValueError("dataset split does not match the locked evaluation config")
    if split is Split.TEST:
        if not allow_frozen_test:
            raise ValueError("frozen test evaluation requires --allow-frozen-test")
        expected = config.get("locked_test_dataset_sha256")
        actual = hashlib.sha256(dataset_bytes).hexdigest()
        if expected != actual:
            raise ValueError("frozen test dataset digest does not match config")
    return split


def _retrieve(examples, corpus, corpus_bytes: bytes, args, config) -> dict[str, list[str]]:
    dense_config, hybrid_config, rerank_config = (
        config["dense"], config["hybrid"], config["reranker"]
    )
    lexical = BM25Index(
        corpus.chunks, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest()
    )
    encoder = TransformersDenseEncoder.from_local_path(
        args.dense_model, model_id=dense_config["id"],
        model_revision=dense_config["revision"], device=dense_config["device"],
        batch_size=int(dense_config["batch_size"]),
    )
    dense = DenseIndex.load(args.dense_index, encoder)
    hybrid = HybridIndex(
        lexical, dense, lexical_index_id=lexical.manifest.index_id,
        dense_index_id=dense.manifest.index_id,
        lexical_weight=float(hybrid_config["lexical_weight"]),
        rrf_constant=int(hybrid_config["rrf_constant"]),
        candidate_k=int(hybrid_config["candidate_k"]),
    )
    scorer = TransformersCrossEncoder.from_local_path(
        args.reranker_model, model_id=rerank_config["id"],
        model_revision=rerank_config["revision"], device=rerank_config["device"],
        batch_size=int(rerank_config["batch_size"]),
    )
    reranked = RerankedIndex(
        hybrid, corpus.chunks, scorer, candidate_index_id=hybrid.manifest.index_id,
        candidate_k=int(rerank_config["candidate_k"]),
    )
    results = {
        example.example_id: [hit.chunk_id for hit in reranked.search(example.question, k=10).hits]
        for example in examples
    }
    del reranked, scorer, hybrid, dense, encoder
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return results


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_outputs(
    output: Path, review_output: Path, identity: dict[str, object],
    scores: list[GenerationExampleResult],
) -> None:
    report = {
        **identity, "aggregate": aggregate_generation(scores),
        "examples": [score.to_dict() for score in scores],
    }
    _write(output, dumps(report))
    _write(review_output, dumps(review_packet(scores)))


def _resume_scores(
    output: Path, identity: dict[str, object],
) -> list[GenerationExampleResult]:
    if not output.exists():
        return []
    previous = json.loads(output.read_text(encoding="utf-8"))
    for key, expected in identity.items():
        if previous.get(key) != expected:
            raise ValueError(f"cannot resume: {key} changed")
    return [GenerationExampleResult.from_dict(value) for value in previous["examples"]]


if __name__ == "__main__":
    raise SystemExit(main())
