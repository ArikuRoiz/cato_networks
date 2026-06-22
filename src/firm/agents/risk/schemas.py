"""Risk agent I/O schemas."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.domain import Portfolio, Trade


class RiskInput(BaseModel):
    proposal: TradeProposal | Hold
    portfolio: Portfolio
    prices: dict[str, Decimal]
    correlation_id: str

    model_config = {"frozen": True}


class ApprovedTrade(BaseModel):
    trade: Trade
    correlation_id: str

    model_config = {"frozen": True}


class HITLRequired(BaseModel):
    proposal: TradeProposal
    reason: str
    correlation_id: str

    model_config = {"frozen": True}


class Rejected(BaseModel):
    reason: str
    correlation_id: str

    model_config = {"frozen": True}
