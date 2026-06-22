"""Trade lifecycle state machine and decision cycle."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from firm.domain.decisions import PolicyResult
    from firm.domain.market import Bar
    from firm.domain.portfolio import Portfolio
    from firm.domain.risk import RiskPolicy


class TradeStatus(StrEnum):
    PROPOSED = "PROPOSED"
    PENDING_HITL = "PENDING_HITL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class Trade(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cycle_id: UUID
    symbol: str
    side: Literal["buy", "sell"]
    qty: Decimal
    status: TradeStatus = TradeStatus.PROPOSED
    requested_price: Decimal
    fill_price: Decimal | None = None
    slippage: Decimal | None = None
    commission: Decimal | None = None
    idempotency_key: str

    model_config = {"frozen": False}

    def revalidate(
        self,
        bar: Bar,
        risk: RiskPolicy,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        start_of_day_nav: Decimal | None = None,
    ) -> PolicyResult:
        """Re-check limits against current bar before execution."""
        current_trade = self.model_copy(update={"requested_price": bar.close})
        return risk.check_trade(current_trade, portfolio, prices, start_of_day_nav)


class DecisionCycle(BaseModel):
    """One end-to-end research → PM → risk → execution cycle."""

    id: UUID = Field(default_factory=uuid4)
    trigger_type: Literal["scheduled", "event"]
    trigger_ref: str | None = None
    started_at: datetime
    outcome: str | None = None

    model_config = {"frozen": False}
