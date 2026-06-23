"""DebaterAgent I/O schemas — one class, two stances."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DebaterInput(BaseModel):
    symbol: str
    round_num: int
    correlation_id: str
    stance: Literal["bull", "bear"]
    evidence_summary: str = ""
    technical_summary: str = ""
    opponent_history: list[str] = []

    model_config = {"frozen": True}


class DebaterCase(BaseModel):
    symbol: str
    round_num: int
    stance: Literal["bull", "bear"]
    argument: str
    key_points: list[str]

    model_config = {"frozen": True}


class DebaterFailure(BaseModel):
    symbol: str
    round_num: int
    stance: Literal["bull", "bear"]
    failure_reason: str

    model_config = {"frozen": True}
