"""Compare RAG and SFT+RAG on evaluation-suite-v2 development data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aerollm.common.documents import Chunk
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.evaluation.sft_comparison import _generate_variant, aggregate
from aerollm.evaluation.task_scoring import validate_task_annotation
from aerollm.retrieval.bm25 import BM25Index
from aerollm.retrieval.dense import DenseIndex
from aerollm.retrieval.dense_transformers import TransformersDenseEncoder
from aerollm.retrieval.hybrid import HybridIndex
from aerollm.retrieval.rerank import RerankedIndex
from aerollm.retrieval.rerank_transformers import TransformersCrossEncoder
from aerollm.retrieval.schemas import RetrievalHit
from aerollm.training.qlora import tree_sha256


def _retrieved_examples(args, examples, corpus, corpus_sha256):  # type: ignore[no-untyped-def]
    lexical = BM25Index(corpus.chunks, corpus_sha256=corpus_sha256)
    encoder = TransformersDenseEncoder.from_local_path(
        args.dense_model,
        model_id="intfloat/e5-small-v2",
        model_revision="f9611f088d69fc3157ff1878217feee72bda0145",
        device="cuda",
        batch_size=32,
    )
    dense = DenseIndex.load(args.dense_index, encoder)
    hybrid = HybridIndex(
        lexical,
        dense,
        lexical_index_id=lexical.manifest.index_id,
        dense_index_id=dense.manifest.index_id,
        lexical_weight=0.25,
        rrf_constant=60,
        candidate_k=100,
    )
    scorer = TransformersCrossEncoder.from_local_path(
        args.reranker_model,
        model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        model_revision="c5ee24cb16019beea0893ab7796b1df96625c6b8",
        device="cuda",
        batch_size=32,
    )
    reranked = RerankedIndex(
        hybrid,
        corpus.chunks,
        scorer,
        candidate_index_id=hybrid.manifest.index_id,
        candidate_k=20,
    )
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    prepared = []
    retrieval_rows = []
    for example in examples:
        hits = reranked.search(example["question"], k=3).hits
        item, row = prepare_retrieved_example(example, hits, chunks)
        prepared.append(item)
        retrieval_rows.append(row)
    del reranked, scorer, hybrid, dense, encoder
    gc.collect()
    return prepared, retrieval_rows


def prepare_retrieved_example(
    example: Mapping[str, Any],
    hits: Sequence[RetrievalHit],
    chunks: Mapping[str, Chunk],
) -> tuple[dict[str, Any], dict[str, object]]:
    """Replace gold evidence with retrieved context and retain retrieval diagnostics."""
    retrieved = [hit.chunk_id for hit in hits]
    item = dict(example)
    item["evidence"] = [
        {"chunk_id": chunk_id, "quote": chunks[chunk_id].text} for chunk_id in retrieved
    ]
    gold = {evidence["chunk_id"] for evidence in example["evidence"]}
    return item, {
        "example_id": example["example_id"],
        "retrieved_chunk_ids": retrieved,
        "gold_hit_at_3": bool(gold.intersection(retrieved)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/v2/development.json"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dense-index", type=Path, required=True)
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    dataset_bytes = args.dataset.read_bytes()
    examples = json.loads(dataset_bytes)["examples"]
    for example in examples:
        validate_task_annotation(example)
    corpus_bytes = args.corpus.read_bytes()
    corpus = CorpusManifest.from_json(corpus_bytes.decode())
    prepared, retrieval_rows = _retrieved_examples(
        args, examples, corpus, hashlib.sha256(corpus_bytes).hexdigest()
    )
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    rag = _generate_variant(
        model,
        tokenizer,
        prepared,
        "rag",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    sft_rag = _generate_variant(
        model,
        tokenizer,
        prepared,
        "sft_rag",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    report = {
        "schema_version": 1,
        "status": "development_retrieved_context_comparison",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "adapter_sha256": tree_sha256(args.adapter),
        "retrieval": {
            "gold_hit_at_3": sum(row["gold_hit_at_3"] for row in retrieval_rows)
            / len(retrieval_rows),
            "examples": retrieval_rows,
        },
        "aggregates": {"rag": aggregate(rag), "sft_rag": aggregate(sft_rag)},
        "predictions": [*rag, *sft_rag],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"retrieval": report["retrieval"]["gold_hit_at_3"], **report["aggregates"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
