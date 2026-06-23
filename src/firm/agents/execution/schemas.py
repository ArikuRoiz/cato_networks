"""Execution agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from firm.agents.risk.schemas import ApprovedTrade
from firm.domain import Portfolio


class ExecutionInput(BaseModel):
    approved_trade: ApprovedTrade
    portfolio_id: UUID
    portfolio: Portfolio
    prices: dict[str, Decimal]
    correlation_id: str
    hitl_approved: bool = False
    """When True the trade was explicitly reviewed and approved by a human
    operator (HITL path).  The execution agent uses ``enforce_hitl_approved``
    instead of ``enforce_before_write`` so the HITL soft-gate is not re-applied
    at the ledger boundary (the human already cleared it)."""

    model_config = {"frozen": True}


class Fill(BaseModel):
    trade_id: UUID
    fill_price: Decimal
    slippage: Decimal
    commission: Decimal
    filled_at: datetime

    model_config = {"frozen": True}


class ExecutionFailure(BaseModel):
    reason: str
    retryable: bool

    model_config = {"frozen": True}
