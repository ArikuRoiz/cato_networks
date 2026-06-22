"""ExecutionAgent — commit an approved trade to the ledger.

Input:  ExecutionInput(approved_trade, portfolio_id, prices, correlation_id)
Output: Fill | ExecutionFailure

Slippage and commission are computed here and stored on the Trade before the
ledger write.  The LedgerGuardrail provides a last-resort policy re-check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from firm.agents.risk import ApprovedTrade
from firm.domain.entities import Portfolio, Trade
from firm.domain.guardrails import LedgerGuardrail, LimitExceeded
from firm.persistence.ledger import LedgerRepository

# Locked constants (SPEC: 5 bps slippage + $0.005/share commission)
_SLIPPAGE_BPS: Decimal = Decimal("0.0005")
_COMMISSION_PER_SHARE: Decimal = Decimal("0.005")


# ---------------------------------------------------------------------------
# I/O schemas
# ---------------------------------------------------------------------------


class ExecutionInput(BaseModel):
    """Input contract for ExecutionAgent."""

    approved_trade: ApprovedTrade
    portfolio_id: UUID
    portfolio: Portfolio
    prices: dict[str, Decimal]
    correlation_id: str

    model_config = {"frozen": True}


class Fill(BaseModel):
    """A successfully executed fill."""

    trade_id: UUID
    fill_price: Decimal
    slippage: Decimal
    commission: Decimal
    filled_at: datetime

    model_config = {"frozen": True}


class ExecutionFailure(BaseModel):
    """Execution could not complete."""

    reason: str
    retryable: bool

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ExecutionAgent:
    """Submit an approved trade to the ledger and return fill details."""

    def __init__(
        self,
        ledger: LedgerRepository,
        guardrail: LedgerGuardrail,
    ) -> None:
        self._ledger = ledger
        self._guardrail = guardrail

    def run(self, inp: ExecutionInput) -> Fill | ExecutionFailure:
        """Execute the trade; return Fill or ExecutionFailure (never raise)."""
        trade = inp.approved_trade.trade
        try:
            self._guardrail.enforce_before_write(trade, inp.portfolio, inp.prices)
        except LimitExceeded as exc:
            return ExecutionFailure(reason=str(exc), retryable=False)

        fill_price = _apply_slippage(trade.requested_price, trade.side)
        commission = _compute_commission(trade.qty)
        filled_trade = _apply_fill_costs(trade, fill_price, commission)

        try:
            return _write_trade(
                self._ledger, filled_trade, inp.portfolio_id, fill_price, commission
            )
        except Exception as exc:
            return ExecutionFailure(reason=str(exc), retryable=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_slippage(price: Decimal, side: str) -> Decimal:
    """Compute fill price with 5-bps slippage in the adverse direction."""
    if side == "buy":
        return price * (Decimal("1") + _SLIPPAGE_BPS)
    return price * (Decimal("1") - _SLIPPAGE_BPS)


def _compute_commission(qty: Decimal) -> Decimal:
    """Fixed $0.005 per share commission."""
    return qty * _COMMISSION_PER_SHARE


def _apply_fill_costs(trade: Trade, fill_price: Decimal, commission: Decimal) -> Trade:
    """Return a trade copy with fill price and commission applied."""
    slippage = abs(fill_price - trade.requested_price) * trade.qty
    return trade.model_copy(
        update={
            "fill_price": fill_price,
            "slippage": slippage,
            "commission": commission,
        }
    )


def _write_trade(
    ledger: LedgerRepository,
    trade: Trade,
    portfolio_id: UUID,
    fill_price: Decimal,
    commission: Decimal,
) -> Fill:
    """Dispatch to ledger.buy or ledger.sell and return a Fill.

    Slippage is already computed and stored on *trade* by ``_apply_fill_costs``;
    do not recalculate here (the trade's fill_price == requested_price at this
    point would produce zero slippage).
    """
    filled_at = datetime.now(tz=UTC)
    if trade.side == "buy":
        ledger.buy(trade, portfolio_id)
    else:
        ledger.sell(trade, portfolio_id)
    slippage = trade.slippage if trade.slippage is not None else Decimal("0")
    return Fill(
        trade_id=trade.id,
        fill_price=fill_price,
        slippage=slippage,
        commission=commission,
        filled_at=filled_at,
    )
