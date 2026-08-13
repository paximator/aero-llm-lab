"""Validated serving configuration loaded without runtime side effects."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ServingConfig:
    service_name: str = "aerollm-serving"
    service_version: str = "v1"
    top_k: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    prompt_version: str = "serving-v1"

    def __post_init__(self) -> None:
        if not self.service_name.strip() or not self.service_version.strip():
            raise ValueError("service name and version are required")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            type(self.temperature) is not float
            or not math.isfinite(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError("temperature must be a non-negative float")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version is required")

    @classmethod
    def from_toml(cls, path: Path) -> ServingConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        if not set(value).issuperset({"service", "answer"}):
            raise ValueError("missing serving configuration tables")
        service = _table(value, "service")
        answer = _table(value, "answer")
        if set(service) != {"name", "version"} or set(answer) != {
            "top_k", "max_new_tokens", "temperature", "prompt_version",
        }:
            raise ValueError("invalid serving configuration fields")
        return cls(
            service_name=service["name"], service_version=service["version"],
            top_k=answer["top_k"], max_new_tokens=answer["max_new_tokens"],
            temperature=answer["temperature"], prompt_version=answer["prompt_version"],
        )  # type: ignore[arg-type]


def _table(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    table = value.get(name)
    if not isinstance(table, Mapping):
        raise ValueError(f"missing [{name}] serving configuration")
    return table
