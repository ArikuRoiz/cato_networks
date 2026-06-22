"""Research agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ResearchInput(BaseModel):
    symbol: str
    decision_ts: datetime
    correlation_id: str

    model_config = {"frozen": True}


class Claim(BaseModel):
    text: str
    source_url: str
    chunk_id: str

    model_config = {"frozen": True}


class Evidence(BaseModel):
    symbol: str
    claims: list[Claim]
    retrieved_at: datetime

    model_config = {"frozen": True}


class Refusal(BaseModel):
    """Research could not proceed — failure is a value, not an exception."""

    reason: Literal[
        "insufficient_evidence",
        "store_unavailable",
        "injection_detected",
        "llm_error_retryable",
        "llm_error_non_retryable",
    ]

    model_config = {"frozen": True}
