"""Portfolio manager agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from firm.agents.research.schemas import Evidence, Refusal
from firm.domain import Portfolio
from firm.domain.enums import TradeSide


class PMInput(BaseModel):
    symbol: str
    evidence: Evidence | Refusal
    portfolio: Portfolio
    decision_ts: datetime
    correlation_id: str
    technical_signal: dict[str, Any] | None = None
    research_plan: dict[str, Any] | None = None

    model_config = {"frozen": True}


class TradeProposal(BaseModel):
    symbol: str
    side: TradeSide
    qty: Decimal
    notional: Decimal
    rationale: str

    model_config = {"frozen": True}


class Hold(BaseModel):
    """Decision not to trade — signal in hold zone or data unavailable."""

    symbol: str
    reason: str

    model_config = {"frozen": True}
