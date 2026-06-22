"""Bull researcher agent I/O schemas."""

from __future__ import annotations

from pydantic import BaseModel


class BullInput(BaseModel):
    symbol: str
    round_num: int
    correlation_id: str
    evidence_summary: str = ""
    technical_summary: str = ""
    bear_history: list[str] = []

    model_config = {"frozen": True}


class BullCase(BaseModel):
    symbol: str
    round_num: int
    argument: str
    key_points: list[str]

    model_config = {"frozen": True}


class BullFailure(BaseModel):
    symbol: str
    round_num: int
    failure_reason: str

    model_config = {"frozen": True}
