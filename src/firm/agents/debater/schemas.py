"""DebaterAgent I/O schemas — one class, two stances."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from firm.agents.research.schemas import Claim


class DebaterInput(BaseModel):
    symbol: str
    round_num: int
    correlation_id: str
    stance: Literal["bull", "bear"]
    evidence_summary: str = ""
    technical_summary: str = ""
    opponent_history: list[str] = []
    # When set (and the agent has an evidence store), the debater runs its own
    # tool-driven search for side-specific evidence before arguing. Optional so
    # the single-shot, no-tools path remains available for tests and fallbacks.
    decision_ts: datetime | None = None

    model_config = {"frozen": True}


class DebaterCase(BaseModel):
    symbol: str
    round_num: int
    stance: Literal["bull", "bear"]
    argument: str
    key_points: list[str]
    # Cited evidence the debater retrieved for its own case. Empty on the
    # single-shot path or a thin-evidence cycle. Reuses the research Claim so
    # grounding (text + chunk_id + source_url) is uniform across agents.
    claims: list[Claim] = []

    model_config = {"frozen": True}


class DebaterFailure(BaseModel):
    symbol: str
    round_num: int
    stance: Literal["bull", "bear"]
    failure_reason: str

    model_config = {"frozen": True}
