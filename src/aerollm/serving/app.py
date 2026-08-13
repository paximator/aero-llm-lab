"""FastAPI application factory for the minimal serving-v1 API."""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import (
    AnswerBackend,
    DependencyInitializer,
    Postprocessor,
    RetrievedPassage,
    Retriever,
    ServingDependencies,
)
from aerollm.serving.fakes import FakeAnswerBackend, FakeRetriever, IdentityPostprocessor
from aerollm.serving.schemas import (
    AnswerRequest,
    AnswerResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    Source,
)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LOGGER = logging.getLogger("aerollm.serving")


class ServiceNotReadyError(RuntimeError):
    pass


class ServingTimeoutError(RuntimeError):
    pass


class RequestTooLargeError(ValueError):
    pass


@dataclass(slots=True)
class _RuntimeState:
    dependencies: ServingDependencies | None
    concurrency: BoundedSemaphore


def create_app(
    *,
    config: ServingConfig | None = None,
    config_path: Path | None = None,
    backend: AnswerBackend | None = None,
    retriever: Retriever | None = None,
    postprocessor: Postprocessor | None = None,
    initializer: DependencyInitializer | None = None,
) -> FastAPI:
    """Create an app with explicit replaceable runtime dependencies.

    Defaults are deterministic fakes so importing or starting the factory never
    allocates model weights. A configured deployment should inject real instances.
    """

    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    settings = config or (ServingConfig.from_toml(config_path) if config_path else ServingConfig())
    injected = (backend, retriever, postprocessor)
    if initializer is not None and any(item is not None for item in injected):
        raise ValueError("initializer cannot be combined with injected dependencies")
    dependencies = None if initializer else ServingDependencies(
        backend=backend or FakeAnswerBackend(),
        retriever=retriever or FakeRetriever(),
        postprocessor=postprocessor or IdentityPostprocessor(),
    )
    runtime = _RuntimeState(dependencies, BoundedSemaphore(settings.max_concurrency))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if initializer is not None:
            try:
                runtime.dependencies = initializer()
            except Exception:
                runtime.dependencies = None
        yield
        runtime.dependencies = None

    app = FastAPI(
        title=settings.service_name, version=settings.service_version, lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request.state.request_id
            return response
        finally:
            _LOGGER.info(json.dumps({
                "event": "http_request",
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status,
                "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            }, sort_keys=True, separators=(",", ":")))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(request, 422, "invalid_request", "Request validation failed")

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _error: Exception) -> JSONResponse:
        return _error_response(request, 500, "internal_error", "Request could not be completed")

    @app.exception_handler(ServiceNotReadyError)
    async def not_ready_error(request: Request, _error: ServiceNotReadyError) -> JSONResponse:
        return _error_response(request, 503, "not_ready", "Service is not ready")

    @app.exception_handler(ServingTimeoutError)
    async def timeout_error(request: Request, _error: ServingTimeoutError) -> JSONResponse:
        return _error_response(request, 504, "timeout", "Request timed out")

    @app.exception_handler(RequestTooLargeError)
    async def too_large_error(request: Request, _error: RequestTooLargeError) -> JSONResponse:
        return _error_response(request, 413, "request_too_large", "Request exceeds service limits")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(service=settings.service_name, version=settings.service_version)

    @app.get(
        "/ready", response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    def ready() -> ReadinessResponse | JSONResponse:
        if runtime.dependencies is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return ReadinessResponse(status="ready")

    @app.post(
        "/v1/answer",
        response_model=AnswerResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
        dependencies = runtime.dependencies
        if dependencies is None:
            raise ServiceNotReadyError
        if len(payload.question) > settings.max_question_characters:
            raise RequestTooLargeError
        started = time.perf_counter()
        try:
            with anyio.fail_after(settings.request_timeout_seconds):
                final_answer, backend_name, passages = await anyio.to_thread.run_sync(
                    _answer_pipeline,
                    dependencies,
                    payload.question,
                    request.state.request_id,
                    settings,
                    runtime.concurrency,
                    abandon_on_cancel=True,
                )
        except TimeoutError as error:
            raise ServingTimeoutError from error
        return AnswerResponse(
            request_id=request.state.request_id,
            answer=final_answer,
            sources=[Source(chunk_id=item.chunk_id, score=item.score) for item in passages],
            backend=backend_name,
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    return app


def _answer_pipeline(
    dependencies: ServingDependencies,
    question: str,
    request_id: str,
    settings: ServingConfig,
    concurrency: BoundedSemaphore,
) -> tuple[str, str, tuple[RetrievedPassage, ...]]:
    with concurrency:
        passages = dependencies.retriever.retrieve(question, top_k=settings.top_k)
        passages = _bounded_context(passages, settings.max_context_characters)
        generated = dependencies.backend.answer(
            question, passages, request_id=request_id,
            max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
            prompt_version=settings.prompt_version,
        )
        final = dependencies.postprocessor.process(generated.text, passages)
        return final, generated.backend, passages


def _bounded_context(
    passages: tuple[RetrievedPassage, ...], limit: int,
) -> tuple[RetrievedPassage, ...]:
    bounded: list[RetrievedPassage] = []
    remaining = limit
    for passage in passages:
        if remaining <= 0:
            break
        text = passage.text[:remaining]
        if text:
            bounded.append(RetrievedPassage(passage.chunk_id, text, passage.score))
            remaining -= len(text)
    return tuple(bounded)


def _error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(error=ErrorDetail(
        code=code, message=message, request_id=request_id,
    ))
    return JSONResponse(status_code=status, content=payload.model_dump())
