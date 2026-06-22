"""Research agent I/O schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from firm.domain.enums import RefusalReason


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

    reason: RefusalReason

    model_config = {"frozen": True}
