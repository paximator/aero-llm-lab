import json

from aerollm.evaluation.runner import PredictionRecord
from aerollm.generation import FakeBackend, GenerationRequest


def test_fake_backend_output_enters_canonical_prediction_contract() -> None:
    output = json.dumps(
        {
            "example_id": "qa-1",
            "output": {
                "answer": "The aircraft landed safely.",
                "citations": [
                    {"chunk_id": "chunk-1", "quote": "landed safely"},
                ],
                "abstained": False,
                "abstention_reason": None,
            },
            "chunks": {"chunk-1": "The aircraft landed safely."},
        }
    )
    backend = FakeBackend(responder=lambda request: output)

    result = backend.generate(
        GenerationRequest(
            prompt="Answer from the supplied evidence.",
            context=("The aircraft landed safely.",),
            request_id="qa-1",
            prompt_version="grounded-qa-v1",
        )
    )
    prediction = PredictionRecord.from_dict(json.loads(result.text))

    assert prediction.example_id == result.trace.request_id
    assert result.trace.identity.backend == "fake"
    assert result.trace.prompt_version == "grounded-qa-v1"
    assert result.trace.context == ("The aircraft landed safely.",)
