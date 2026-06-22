"""LLM-as-a-judge agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class JudgeInput(BaseModel):
    symbol: str
    decision_ts: datetime
    correlation_id: str
    evidence: dict[str, Any] | None = None
    technical_signal: dict[str, Any] | None = None
    trade_proposal: dict[str, Any] | None = None
    cycle_outcome: str | None = None
    synthesis: dict[str, Any] | None = None

    model_config = {"frozen": True}


class Verdict(BaseModel):
    """Independent quality assessment of a full decision cycle.

    ``coherence_score`` is 1 (highly inconsistent) to 5 (fully coherent).
    ``flags`` are specific, actionable observations — not generic commentary.
    """

    correlation_id: str
    coherence_score: int  # 1-5
    alignment: Literal["aligned", "partial", "misaligned"]
    flags: list[str]
    recommendation: str
    reasoning: str

    model_config = {"frozen": True}
