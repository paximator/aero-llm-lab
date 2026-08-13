"""FastAPI application factory for the minimal serving-v1 API."""

from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import (
    AnswerBackend,
    DependencyInitializer,
    Postprocessor,
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


class ServiceNotReadyError(RuntimeError):
    pass


@dataclass(slots=True)
class _RuntimeState:
    dependencies: ServingDependencies | None


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
    runtime = _RuntimeState(dependencies)

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
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(request, 422, "invalid_request", "Request validation failed")

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _error: Exception) -> JSONResponse:
        return _error_response(request, 500, "internal_error", "Request could not be completed")

    @app.exception_handler(ServiceNotReadyError)
    async def not_ready_error(request: Request, _error: ServiceNotReadyError) -> JSONResponse:
        return _error_response(request, 503, "not_ready", "Service is not ready")

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
    def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
        dependencies = runtime.dependencies
        if dependencies is None:
            raise ServiceNotReadyError
        started = time.perf_counter()
        passages = dependencies.retriever.retrieve(payload.question, top_k=settings.top_k)
        generated = dependencies.backend.answer(
            payload.question, passages, request_id=request.state.request_id,
            max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
            prompt_version=settings.prompt_version,
        )
        final_answer = dependencies.postprocessor.process(generated.text, passages)
        return AnswerResponse(
            request_id=request.state.request_id,
            answer=final_answer,
            sources=[Source(chunk_id=item.chunk_id, score=item.score) for item in passages],
            backend=generated.backend,
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    return app


def _error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(error=ErrorDetail(
        code=code, message=message, request_id=request_id,
    ))
    return JSONResponse(status_code=status, content=payload.model_dump())
