"""ExecutionAgent — commit an approved trade to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from firm.agents.base import BaseAgent
from firm.agents.execution.schemas import ExecutionFailure, ExecutionInput, Fill
from firm.domain import Trade
from firm.domain.guardrails import LedgerGuardrail, LimitExceeded
from firm.persistence.ledger import LedgerRepository

# Locked constants (5 bps slippage + $0.005/share commission)
_SLIPPAGE_BPS: Decimal = Decimal("0.0005")
_COMMISSION_PER_SHARE: Decimal = Decimal("0.005")


class ExecutionAgent(BaseAgent[ExecutionInput, Fill | ExecutionFailure]):
    def __init__(self, ledger: LedgerRepository, guardrail: LedgerGuardrail) -> None:
        self._ledger = ledger
        self._guardrail = guardrail

    def run(self, inp: ExecutionInput) -> Fill | ExecutionFailure:
        trade = inp.approved_trade.trade
        try:
            self._guardrail.enforce_before_write(trade, inp.portfolio, inp.prices)
        except LimitExceeded as exc:
            return ExecutionFailure(reason=str(exc), retryable=False)

        fill_price = _apply_slippage(trade.requested_price, trade.side)
        commission = _compute_commission(trade.qty)
        filled_trade = _apply_fill_costs(trade, fill_price, commission)

        try:
            return _write_trade(self._ledger, filled_trade, inp.portfolio_id, fill_price, commission)
        except Exception as exc:
            return ExecutionFailure(reason=str(exc), retryable=True)


def _apply_slippage(price: Decimal, side: str) -> Decimal:
    if side == "buy":
        return price * (Decimal("1") + _SLIPPAGE_BPS)
    return price * (Decimal("1") - _SLIPPAGE_BPS)


def _compute_commission(qty: Decimal) -> Decimal:
    return qty * _COMMISSION_PER_SHARE


def _apply_fill_costs(trade: Trade, fill_price: Decimal, commission: Decimal) -> Trade:
    slippage = abs(fill_price - trade.requested_price) * trade.qty
    return trade.model_copy(
        update={"fill_price": fill_price, "slippage": slippage, "commission": commission}
    )


def _write_trade(
    ledger: LedgerRepository,
    trade: Trade,
    portfolio_id: object,
    fill_price: Decimal,
    commission: Decimal,
) -> Fill:
    filled_at = datetime.now(tz=UTC)
    if trade.side == "buy":
        ledger.buy(trade, portfolio_id)  # type: ignore[arg-type]
    else:
        ledger.sell(trade, portfolio_id)  # type: ignore[arg-type]
    slippage = trade.slippage if trade.slippage is not None else Decimal("0")
    return Fill(
        trade_id=trade.id,
        fill_price=fill_price,
        slippage=slippage,
        commission=commission,
        filled_at=filled_at,
    )
