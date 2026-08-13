import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi.testclient import TestClient

from aerollm.serving.app import create_app
from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import BackendAnswer, RetrievedPassage, ServingDependencies
from aerollm.serving.fakes import FakeAnswerBackend, FakeRetriever, IdentityPostprocessor


def test_health_reports_configured_service() -> None:
    client = TestClient(create_app(config=ServingConfig(
        service_name="test-serving", service_version="v1",
    )))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "test-serving", "version": "v1"}
    assert re.fullmatch(r"[0-9a-f-]{36}", response.headers["x-request-id"])


def test_answer_is_deterministic_and_propagates_request_id() -> None:
    client = TestClient(create_app())

    first = client.post(
        "/v1/answer", json={"question": "What happened?"},
        headers={"x-request-id": "request-123"},
    )
    second = client.post("/v1/answer", json={"question": "What happened?"})

    assert first.status_code == 200
    assert first.headers["x-request-id"] == "request-123"
    assert first.json()["request_id"] == "request-123"
    assert first.json()["answer"] == second.json()["answer"]
    assert first.json()["backend"] == "deterministic-fake"
    assert len(first.json()["sources"]) == 2
    assert first.json()["latency_ms"] >= 0


def test_dependencies_are_injected_through_protocol_boundaries() -> None:
    events: list[object] = []

    class Retriever:
        def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]:
            events.append((query, top_k))
            return (RetrievedPassage("chunk-7", "evidence", 0.75),)

    class Backend:
        def answer(self, question, passages, **options):  # type: ignore[no-untyped-def]
            events.append((question, passages, options))
            return BackendAnswer(" raw ", "injected")

    class Postprocessor:
        def process(self, answer: str, passages: tuple[RetrievedPassage, ...]) -> str:
            events.append((answer, passages))
            return answer.strip()

    client = TestClient(create_app(
        retriever=Retriever(), backend=Backend(), postprocessor=Postprocessor(),
    ))
    response = client.post("/v1/answer", json={"question": "Question"})

    assert response.json()["answer"] == "raw"
    assert response.json()["sources"] == [{"chunk_id": "chunk-7", "score": 0.75}]
    assert len(events) == 3


def test_validation_and_backend_failures_do_not_leak_details() -> None:
    invalid = TestClient(create_app()).post("/v1/answer", json={"question": " "})

    class FailingRetriever:
        def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]:
            raise RuntimeError("secret model path")

    failing_client = TestClient(
        create_app(retriever=FailingRetriever()), raise_server_exceptions=False,
    )
    failed = failing_client.post(
        "/v1/answer", json={"question": "valid"}, headers={"x-request-id": "failure-1"},
    )

    assert invalid.status_code == 422
    assert invalid.json()["error"]["message"] == "Request validation failed"
    assert failed.status_code == 500
    assert failed.json() == {"error": {
        "code": "internal_error", "message": "Request could not be completed",
        "request_id": "failure-1",
    }}
    assert "secret" not in failed.text


def test_invalid_incoming_request_id_is_replaced() -> None:
    response = TestClient(create_app()).get("/health", headers={"x-request-id": "bad id"})

    assert response.headers["x-request-id"] != "bad id"


def test_static_dependencies_are_ready() -> None:
    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_initializer_runs_at_startup_and_becomes_ready() -> None:
    calls: list[str] = []

    def initialize() -> ServingDependencies:
        calls.append("initialize")
        return ServingDependencies(
            FakeAnswerBackend(), FakeRetriever(), IdentityPostprocessor(),
        )

    app = create_app(initializer=initialize)
    with TestClient(app) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.post("/v1/answer", json={"question": "Question"}).status_code == 200

    assert calls == ["initialize"]


def test_failed_initializer_stays_live_but_unready_without_leaking_error() -> None:
    def initialize() -> ServingDependencies:
        raise RuntimeError("secret local model path")

    app = create_app(initializer=initialize)
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        readiness = client.get("/ready")
        answer = client.post(
            "/v1/answer", json={"question": "Question"},
            headers={"x-request-id": "not-ready-1"},
        )

    assert health.status_code == 200
    assert readiness.status_code == 503
    assert readiness.json() == {"status": "not_ready"}
    assert answer.status_code == 503
    assert answer.json()["error"] == {
        "code": "not_ready", "message": "Service is not ready",
        "request_id": "not-ready-1",
    }
    assert "secret" not in answer.text


def test_question_and_context_limits_are_enforced() -> None:
    observed: list[str] = []

    class Backend:
        def answer(self, question, passages, **options):  # type: ignore[no-untyped-def]
            del question, options
            observed.extend(item.text for item in passages)
            return BackendAnswer("answer", "bounded")

    class Retriever:
        def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedPassage, ...]:
            del query, top_k
            return (
                RetrievedPassage("one", "abcdef", 1.0),
                RetrievedPassage("two", "second", 0.5),
            )

    config = ServingConfig(max_question_characters=8, max_context_characters=5)
    client = TestClient(create_app(config=config, backend=Backend(), retriever=Retriever()))

    accepted = client.post("/v1/answer", json={"question": "question"})
    rejected = client.post("/v1/answer", json={"question": "too long!"})

    assert accepted.status_code == 200
    assert observed == ["abcde"]
    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "request_too_large"


def test_slow_backend_returns_safe_timeout() -> None:
    class Backend:
        def answer(self, question, passages, **options):  # type: ignore[no-untyped-def]
            del question, passages, options
            time.sleep(0.05)
            return BackendAnswer("late", "slow")

    config = ServingConfig(request_timeout_seconds=0.01)
    response = TestClient(create_app(config=config, backend=Backend())).post(
        "/v1/answer", json={"question": "Question"}, headers={"x-request-id": "timeout-1"},
    )

    assert response.status_code == 504
    assert response.json()["error"] == {
        "code": "timeout", "message": "Request timed out", "request_id": "timeout-1",
    }


def test_backend_concurrency_is_bounded() -> None:
    lock = Lock()
    active = 0
    maximum_active = 0

    class Backend:
        def answer(self, question, passages, **options):  # type: ignore[no-untyped-def]
            nonlocal active, maximum_active
            del question, passages, options
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return BackendAnswer("answer", "bounded")

    config = ServingConfig(max_concurrency=1, request_timeout_seconds=1.0)
    client = TestClient(create_app(config=config, backend=Backend()))

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(
            lambda _: client.post("/v1/answer", json={"question": "Question"}), range(2),
        ))

    assert [response.status_code for response in responses] == [200, 200]
    assert maximum_active == 1


def test_request_log_is_structured_and_excludes_question(caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="aerollm.serving")

    response = TestClient(create_app()).post(
        "/v1/answer", json={"question": "sensitive question"},
        headers={"x-request-id": "log-1"},
    )

    record = json.loads(caplog.records[-1].message)
    assert response.status_code == 200
    assert record == {
        "event": "http_request", "request_id": "log-1", "method": "POST",
        "path": "/v1/answer", "status_code": 200,
        "latency_ms": record["latency_ms"],
    }
    assert record["latency_ms"] >= 0
    assert "sensitive" not in caplog.records[-1].message
