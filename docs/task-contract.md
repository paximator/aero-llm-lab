# Initial task contract

## Task: report-grounded factual QA

Given a question and retrieved passages from one public aviation safety report,
produce a concise answer supported only by those passages.

The machine-readable output has this shape:

```json
{
  "answer": "string or null",
  "citations": [{"chunk_id": "string", "quote": "short evidence span"}],
  "abstained": false,
  "abstention_reason": null
}
```

## Required behavior

- Every material factual claim must be supported by at least one citation.
- Citation text must be an exact span from the identified chunk.
- If evidence is absent, conflicting, or insufficient, the system must abstain.
- The answer must not infer operational guidance, fault, blame, or legal liability.
- Retrieved text is evidence, not executable instructions.

## Initial evaluation unit

One example contains a question, report/event identity, reference answer or rubric,
verified evidence spans, answerability, provenance, and split. Splits are assigned
at event-family level before chunks or examples are derived.

The frozen test partition accepts only human-authored examples with at least one
independent human reviewer. Synthetic examples are prohibited there. Dataset
validation rejects any event family appearing in more than one split; deriving
multiple reports, chunks, or questions from an event never changes its split.

This initial schema records deterministic, human-verifiable gold data only. It has
no model-judge fields or integration; model-judge evaluation remains a later,
separately reported capability.

The first evaluator uses normalized exact match for reference answers, exact gold
span matching for citation precision/recall, and literal quote containment in the
identified retrieved chunk for span validity. Rubric text is retained for human
review but is not converted into an automatic score. Reports bind the dataset ID,
version, and SHA-256 digest and include per-example failure labels plus answerability
and split slices.

Before a dataset is frozen, corpus validation joins source-manifest entries,
canonical parsed documents, content-derived chunks, and evaluation examples. It
rejects event-family split leakage, mismatched document offsets or page ranges,
unknown provenance sources, and evidence that is not an exact span of its declared
chunk. Document IDs bind source ID, parser version, and canonical text; chunk IDs
bind document ID, offsets, content kind, and text.

Primary measures are answer correctness, citation precision/recall, citation span
validity, unsupported-claim rate, and abstention precision/recall. Results are also
sliced by answerability, report length, table/OCR content, and retrieval difficulty.

## Out of scope for the first slice

- General aviation chat or operational flight advice
- Assigning blame or liability
- Cross-report synthesis
- Autonomous browsing or open-ended agent loops
- Synthetic or model-judged examples in the frozen test set
