# Retrieval test set v1

## Outcome

Retrieval test set `ntsb-retrieval-test` version `1.0.0` is frozen with 10
independently reviewed, answerable examples from the held-out NTSB event family
`DCA25MA108`. Two examples require multiple evidence spans. All frozen spans resolve
to exact text in the versioned corpus, including PDF layout whitespace.

## Reproducibility

- Dataset: `data/evaluation/retrieval_test_v1.json`
- Dataset SHA-256: `c095f7bd656f0dfebfdd7ea4cb221c7a11d76c7a6aef1df3ebd9cf952e044723`
- Corpus: `artifacts/corpora/ntsb-pilot-v1.json`
- Corpus SHA-256: `37aa9ed77c3476b149ed8f4d455a7e8680e9c0359edf323117c422d4a765ac6d`
- Corpus version: `ntsb-pilot-v1`
- Corpus scope: 16 documents and 6,897 chunks
- Reviewer: `github:paximator`

```powershell
uv run aerollm-freeze-retrieval-test
uv run aerollm-validate-corpus artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_test_v1.json
```

The freeze command is idempotent for identical bytes and refuses to overwrite a
different frozen artifact. Corrections require a new dataset version and output path.

## Leakage policy

The complete `DCA25MA108` event family is assigned to the test split. Its questions,
answers, evidence spans, retrieval failures, and test metrics must not be used for
training, prompt selection, retrieval or reranking design, model selection, or
hyperparameter tuning. Development-split results drive those decisions. The test set
is used only for locked milestone measurements.

## Limitations

- The first version contains only 10 examples from one event family.
- Every example is answerable; abstention coverage must be added in a future version.
- Exact frozen evidence preserves PDF extraction artifacts, while the review packet
  retains the cleaned human-readable form.
