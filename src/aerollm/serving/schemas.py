"""Public serving-v1 request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=8_000)
    event_id: str | None = Field(default=None, min_length=1, max_length=128)
    report_id: str | None = Field(default=None, min_length=1, max_length=256)


class Source(BaseModel):
    chunk_id: str
    score: float
    event_id: str | None = None
    report_id: str | None = None


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    backend: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
