"""ExecutionAgent — commit an approved trade to the ledger."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

from firm.agents.base import BaseAgent
from firm.agents.execution.schemas import ExecutionFailure, ExecutionInput, Fill
from firm.domain.enums import TradeSide
from firm.domain.guardrails import LedgerGuardrail, LimitExceeded
from firm.persistence.ledger import LedgerRepository

logger = logging.getLogger(__name__)


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

        try:
            if trade.side == TradeSide.BUY:
                filled = self._ledger.buy(trade, inp.portfolio_id)
            else:
                filled = self._ledger.sell(trade, inp.portfolio_id)
        except Exception as exc:
            logger.exception("Ledger write failed for trade %s", trade.id)
            return ExecutionFailure(reason=str(exc), retryable=True)

        return Fill(
            trade_id=filled.id,
            fill_price=filled.fill_price or trade.requested_price,
            slippage=filled.slippage or Decimal("0"),
            commission=filled.commission or Decimal("0"),
            filled_at=datetime.now(tz=UTC),
        )
