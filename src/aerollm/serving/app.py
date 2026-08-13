"""FastAPI application factory for the minimal serving-v1 API."""

from __future__ import annotations

import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aerollm.serving.config import ServingConfig
from aerollm.serving.contracts import AnswerBackend, Postprocessor, Retriever
from aerollm.serving.fakes import FakeAnswerBackend, FakeRetriever, IdentityPostprocessor
from aerollm.serving.schemas import (
    AnswerRequest,
    AnswerResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    Source,
)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def create_app(
    *,
    config: ServingConfig | None = None,
    config_path: Path | None = None,
    backend: AnswerBackend | None = None,
    retriever: Retriever | None = None,
    postprocessor: Postprocessor | None = None,
) -> FastAPI:
    """Create an app with explicit replaceable runtime dependencies.

    Defaults are deterministic fakes so importing or starting the factory never
    allocates model weights. A configured deployment should inject real instances.
    """

    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    settings = config or (ServingConfig.from_toml(config_path) if config_path else ServingConfig())
    answer_backend = backend or FakeAnswerBackend()
    passage_retriever = retriever or FakeRetriever()
    answer_postprocessor = postprocessor or IdentityPostprocessor()
    app = FastAPI(title=settings.service_name, version=settings.service_version)

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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(service=settings.service_name, version=settings.service_version)

    @app.post(
        "/v1/answer",
        response_model=AnswerResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
        started = time.perf_counter()
        passages = passage_retriever.retrieve(payload.question, top_k=settings.top_k)
        generated = answer_backend.answer(
            payload.question, passages, request_id=request.state.request_id,
            max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
            prompt_version=settings.prompt_version,
        )
        final_answer = answer_postprocessor.process(generated.text, passages)
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
