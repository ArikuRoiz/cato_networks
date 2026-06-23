"""LLM-as-a-judge agent I/O schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from firm.agents.synthesis.schemas import CycleSnapshot
from firm.domain.enums import VerdictAlignment


class JudgeInput(CycleSnapshot):
    """Cycle snapshot plus the synthesis memo the judge must evaluate."""

    synthesis: dict[str, Any] | None = None


class Verdict(BaseModel):
    """Independent quality assessment of a full decision cycle.

    ``coherence_score`` is 1 (highly inconsistent) to 5 (fully coherent).
    ``flags`` are specific, actionable observations — not generic commentary.
    """

    correlation_id: str
    coherence_score: int = Field(ge=1, le=5)
    alignment: VerdictAlignment
    flags: list[str]
    recommendation: str
    reasoning: str

    model_config = {"frozen": True}


class JudgeFailure(BaseModel):
    correlation_id: str
    failure_reason: str

    model_config = {"frozen": True}
