"""LLM request/response value objects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel


class ToolDef(BaseModel):
    """Specification for a tool the LLM can call."""

    name: str
    description: str
    input_schema: dict[str, Any]

    model_config = {"frozen": True}


# Maps tool name → callable that executes it and returns a plain-text result.
ToolExecutors = dict[str, Callable[[dict[str, Any]], str]]


class LLMMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str

    model_config = {"frozen": True}


class LLMError(BaseModel):
    """A failed LLM call — retryable or terminal."""

    message: str
    retryable: bool

    model_config = {"frozen": True}
