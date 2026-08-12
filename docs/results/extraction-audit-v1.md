# Extraction and chunk-label audit v1

Status: complete; one safe correction applied and one field quarantined.

## Objective

Check whether parsed text is usable for evidence retrieval, determine why six tiny
chunks passed through the pipeline, and estimate whether `Chunk.kind` is reliable
enough to use as a retrieval feature.

## Method

The label sample was deterministic: sort chunks by the SHA-256 of `chunk_id`, then
inspect the first 15 chunks labeled `table` and the first 15 labeled `text`. The
review checked visible extraction continuity, page provenance, whether layout
preserved meaningful structure, and whether the assigned kind matched the content.
All six chunks below the configured 100-character warning threshold were inspected
separately.

## Findings

| Check | Result |
|---|---:|
| Stratified chunks inspected | 30 |
| Correct `text` labels | 15 / 15 |
| Correct `table` or structured-layout labels | 7 / 15 |
| Observed precision of `table` label | 46.7% |
| Tiny chunks inspected | 6 / 6 |
| Tiny chunks containing only repeated `NTSB` header | 6 / 6 |

The inspected prose remains understandable and suitable for lexical retrieval, but
layout extraction introduces excess spaces, occasional token concatenation, and
running headers. True structured examples included forecast tables, communication
transcripts, and tables of contents. False table labels were ordinary paragraphs
whose justified/layout-positioned text contained multiple spaces.

Sample judgments by stable ID prefix:

| Predicted | Correct | Chunk ID prefixes |
|---|---:|---|
| `table` | Yes | `b62ee32a`, `e1805bf9`, `94280256`, `55e3d30b`, `464f0b28`, `cf6781cc`, `42f0ffc2` |
| `table` | No | `e08d4df0`, `d11f4740`, `27c02887`, `46ca07fe`, `918e5ad1`, `a5e23428`, `09d6c6f8`, `e585968a` |
| `text` | Yes | `838fb2a5`, `20756af6`, `dcfe2578`, `5ea8bd8d`, `f9b32340`, `4adc8ca5`, `6cbf1b7c`, `3542ca2e`, `e8727beb`, `006b7f29`, `ea903f1b`, `ef713bb1`, `d7d7178f`, `f5495399`, `34eb82f7` |

## Correction

The chunker now rejects a page whose only extracted content is the repeated `NTSB`
header. This rule is intentionally narrow and regression-tested. The corpus was
rechunked and rebuilt; validation warnings fell from six to zero.

No replacement table classifier was introduced. Thirty samples establish that the
existing heuristic is unreliable, but do not justify a robust new classifier.
`kind` remains serialized for compatibility and diagnosis but is quarantined from
retrieval features and evaluation stratification.

## Limitations and decision

This is a small deterministic audit, not an estimate with narrow statistical
uncertainty. It did not compare extraction against page images or audit every
document. OCR-specific quality is not assessed because this corpus uses embedded
PDF text.

Decision: proceed to evidence-set authoring and BM25 using chunk text and provenance
only. Add a larger page-image audit or a dedicated layout parser before making any
claim about table-aware retrieval.

