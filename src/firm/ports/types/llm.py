"""LLM request/response value objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
