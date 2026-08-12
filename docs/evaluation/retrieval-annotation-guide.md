# Retrieval evaluation annotation guide

## Purpose

Create questions that test whether retrieval finds the report passage needed to
answer, independently of generation quality. Development examples may be drafted
with model assistance. Frozen test examples require an independent human reviewer.

## Authoring rules

1. Write the question without copying distinctive wording from the evidence unless
   the task intentionally tests identifier lookup.
2. Link every answerable question to the smallest exact evidence phrase and its
   stable chunk ID.
3. Keep all evidence within the example's report and event split.
4. Include factual, causal, temporal, mechanical, weather, recommendation, and
   negative/abstention questions.
5. For an unanswerable item, search the complete report and state why the requested
   information is not established.
6. Reject ambiguous questions with multiple conflicting answers or evidence that
   requires unsupported inference.
7. Do not tune retrieval using test questions, test evidence, or test failures.

## Review checklist

- The question is answerable from the cited phrase, or genuinely absent.
- The reference answer does not add unsupported details.
- The quote is exact despite PDF layout whitespace.
- Page, chunk, report, event family, and split are correct.
- The question is not a paraphrased duplicate.
- A test reviewer is independent of the author and records their identity.

## Commands

```powershell
uv run aerollm-build-retrieval-dataset
uv run aerollm-retrieve artifacts/corpora/ntsb-pilot-v1.json `
  data/evaluation/retrieval_dev_v1.json --k 10 `
  --json-output artifacts/runs/bm25-dev-v1/report.json `
  --markdown-output artifacts/runs/bm25-dev-v1/report.md
```

The current development set is model-assisted and must be manually sampled before
its metrics are used in portfolio claims. The test split remains unscored until its
10 drafts receive independent human review.

