"""Synthesis report agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SynthesisInput(BaseModel):
    symbol: str
    decision_ts: datetime
    correlation_id: str
    evidence: dict[str, Any] | None = None
    technical_signal: dict[str, Any] | None = None
    research_plan: dict[str, Any] | None = None
    trade_proposal: dict[str, Any] | None = None
    cycle_outcome: str | None = None

    model_config = {"frozen": True}


class SynthesisReport(BaseModel):
    """LLM-written investment memo covering the full decision cycle."""

    symbol: str
    correlation_id: str
    title: str
    executive_summary: str
    evidence_synthesis: str
    decision_rationale: str
    execution_quality: str

    model_config = {"frozen": True}


class SynthesisFailure(BaseModel):
    reason: str

    model_config = {"frozen": True}
