"""Portfolio aggregate — cash, lots, and holdings."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from firm.domain.exceptions import InsufficientCash, InsufficientHolding

# Shared trade-cost constants (also used by persistence/ledger.py).
_COMMISSION_PER_SHARE: Decimal = Decimal("0.005")
_SLIPPAGE_BPS: Decimal = Decimal("0.0005")  # 5 bps


class Lot(BaseModel):
    """One FIFO cost-basis layer, mutable qty."""

    id: UUID = Field(default_factory=uuid4)
    symbol: str
    qty: Decimal
    cost: Decimal  # per-share cost
    opened_at: datetime

    model_config = {"frozen": False}


class Holding(BaseModel):
    """Aggregates lots for one symbol."""

    symbol: str
    lots: list[Lot] = Field(default_factory=list)

    model_config = {"frozen": False}

    @property
    def quantity(self) -> Decimal:
        return sum((lot.qty for lot in self.lots), Decimal("0"))

    @property
    def avg_cost(self) -> Decimal:
        total_qty = self.quantity
        if total_qty == Decimal("0"):
            return Decimal("0")
        total_cost = sum((lot.qty * lot.cost for lot in self.lots), Decimal("0"))
        return total_cost / total_qty

    def open_lots(self) -> list[Lot]:
        return [lot for lot in self.lots if lot.qty > Decimal("0")]

    def close_lots(self, qty: Decimal) -> list[tuple[Lot, Decimal]]:
        """FIFO — consume lots oldest-first until *qty* is satisfied.

        Raises ``InsufficientHolding`` when requested qty > open position.
        """
        if qty > self.quantity:
            raise InsufficientHolding(
                f"Cannot close {qty} of {self.symbol}; only {self.quantity} held."
            )
        sorted_lots = sorted(self.open_lots(), key=lambda lot: lot.opened_at)
        return _consume_fifo(sorted_lots, qty)


def _consume_fifo(
    lots: list[Lot],
    remaining: Decimal,
) -> list[tuple[Lot, Decimal]]:
    result: list[tuple[Lot, Decimal]] = []
    for lot in lots:
        if remaining <= Decimal("0"):
            break
        consumed = min(lot.qty, remaining)
        result.append((lot, consumed))
        remaining -= consumed
    return result


class Portfolio(BaseModel):
    """Cash + holdings aggregate, NAV arithmetic."""

    id: UUID = Field(default_factory=uuid4)
    cash: Decimal
    holdings: dict[str, Holding] = Field(default_factory=dict)

    model_config = {"frozen": False}

    def can_afford(self, symbol: str, qty: Decimal, price: Decimal) -> bool:
        """True when cash covers notional + per-share commission.

        Raises ``InsufficientCash`` when the portfolio cannot fund the trade.
        """
        commission = qty * _COMMISSION_PER_SHARE
        required = qty * price + commission
        if self.cash < required:
            raise InsufficientCash(
                f"Cannot afford {qty} {symbol} @ {price}; "
                f"required {required}, available {self.cash}."
            )
        return True

    def nav(self, prices: dict[str, Decimal]) -> Decimal:
        """Net asset value: cash plus market value of all holdings."""
        missing = self.holdings.keys() - prices.keys()
        if missing:
            raise ValueError(f"prices dict is missing symbols required for NAV: {sorted(missing)}")
        equity = sum(holding.quantity * prices[symbol] for symbol, holding in self.holdings.items())
        return self.cash + equity

    def position_weight(self, symbol: str, prices: dict[str, Decimal]) -> Decimal:
        """Fraction of NAV represented by *symbol* (0 if not held)."""
        nav = self.nav(prices)
        if nav == Decimal("0"):
            return Decimal("0")
        holding = self.holdings.get(symbol)
        if holding is None:
            return Decimal("0")
        return holding.quantity * prices[symbol] / nav
