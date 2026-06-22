"""Pure domain entities — zero IO imports.

All monetary values use ``Decimal``; float is forbidden.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# DecisionCycle — one complete research→PM→risk→execution cycle
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMMISSION_PER_SHARE: Decimal = Decimal("0.005")
_SLIPPAGE_BPS: Decimal = Decimal("0.0005")  # 5 bps


# ---------------------------------------------------------------------------
# Custom domain exceptions — no framework imports
# ---------------------------------------------------------------------------


class InsufficientHolding(Exception):
    """Raised when a close request exceeds the open position."""


class InsufficientCash(Exception):
    """Raised when cash is insufficient to fund a trade."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TradeStatus(StrEnum):
    PROPOSED = "PROPOSED"
    PENDING_HITL = "PENDING_HITL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# PolicyResult union — result types, not exceptions
# ---------------------------------------------------------------------------


class Approved(BaseModel):
    """Trade passed all RiskPolicy checks."""

    model_config = {"frozen": True}


class HITLRequired(BaseModel):
    """Trade requires human-in-the-loop approval before execution."""

    reason: str

    model_config = {"frozen": True}


class Rejected(BaseModel):
    """Trade was hard-rejected by RiskPolicy."""

    reason: str

    model_config = {"frozen": True}


PolicyResult = Approved | HITLRequired | Rejected


# ---------------------------------------------------------------------------
# Bar — single OHLCV candle, immutable
# ---------------------------------------------------------------------------


class Bar(BaseModel):
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    ts: datetime

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Lot — one FIFO cost-basis layer, mutable qty
# ---------------------------------------------------------------------------


class Lot(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    qty: Decimal
    cost: Decimal  # per-share cost
    opened_at: datetime

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Holding — aggregates lots for one symbol
# ---------------------------------------------------------------------------


class Holding(BaseModel):
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

        Returns list of ``(lot, consumed_qty)`` pairs for ledger writes.
        Raises ``InsufficientHolding`` when requested qty > open position.

        Lots are sorted by ``opened_at`` regardless of list-insertion order so
        FIFO correctness holds even when lots arrive from an unordered source
        (e.g., a DB query without ORDER BY).
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
    """Peel off lots oldest-first until *remaining* is consumed."""
    result: list[tuple[Lot, Decimal]] = []
    for lot in lots:
        if remaining <= Decimal("0"):
            break
        consumed = min(lot.qty, remaining)
        result.append((lot, consumed))
        remaining -= consumed
    return result


# ---------------------------------------------------------------------------
# Portfolio — cash + holdings, NAV arithmetic
# ---------------------------------------------------------------------------


class Portfolio(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cash: Decimal
    holdings: dict[str, Holding] = Field(default_factory=dict)

    model_config = {"frozen": False}

    def can_afford(self, symbol: str, qty: Decimal, price: Decimal) -> bool:
        """True when cash covers notional + per-share commission.

        Raises ``InsufficientCash`` when the portfolio cannot fund the trade,
        consistent with ``close_lots`` raising ``InsufficientHolding``.
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
        """Net asset value: cash plus market value of all holdings.

        Raises ``ValueError`` for any held symbol missing from *prices* so
        callers get a domain-meaningful error instead of a bare ``KeyError``.
        """
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


# ---------------------------------------------------------------------------
# RiskPolicy — limit enforcement, returns a PolicyResult
# ---------------------------------------------------------------------------


class RiskPolicy(BaseModel):
    max_trade_notional_pct: Decimal  # e.g. Decimal("0.10") → 10 % of NAV
    max_name_concentration_pct: Decimal  # e.g. Decimal("0.25") → 25 % of NAV
    daily_loss_halt_pct: Decimal  # e.g. Decimal("0.03") -> -3 % daily NAV
    hitl_threshold_pct: Decimal  # e.g. Decimal("0.05") → 5 % of NAV

    model_config = {"frozen": True}

    def check_daily_halt(
        self,
        current_nav: Decimal,
        start_of_day_nav: Decimal,
    ) -> PolicyResult:
        """Return ``Rejected`` if intraday loss has breached the daily halt threshold.

        The daily-loss halt is a LOCKED DECISION: no new trades are permitted
        once the portfolio has lost more than ``daily_loss_halt_pct`` of its
        opening NAV.  Returns ``Approved`` when the threshold is not breached.
        """
        if start_of_day_nav == Decimal("0"):
            return Rejected(reason="Start-of-day NAV is zero; cannot evaluate daily halt.")
        loss_pct = (start_of_day_nav - current_nav) / start_of_day_nav
        if loss_pct >= self.daily_loss_halt_pct:
            return Rejected(
                reason=(
                    f"Daily loss {loss_pct:.2%} has reached or exceeded halt "
                    f"threshold -{self.daily_loss_halt_pct:.2%}; trading halted."
                )
            )
        return Approved()

    def check_trade(
        self,
        trade: Trade,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        start_of_day_nav: Decimal | None = None,
    ) -> PolicyResult:
        """Evaluate all risk limits for *trade*.

        Pass ``start_of_day_nav`` to enforce the -3 % daily-loss halt.
        When omitted the daily-halt check is skipped (useful in tests that
        focus on per-trade limits only).
        """
        nav = portfolio.nav(prices)
        if nav == Decimal("0"):
            return Rejected(reason="Portfolio NAV is zero; cannot size trade.")
        if start_of_day_nav is not None:
            halt = self.check_daily_halt(nav, start_of_day_nav)
            if isinstance(halt, Rejected):
                return halt
        notional = trade.qty * trade.requested_price
        return _evaluate_trade_limits(self, trade, notional, nav, portfolio, prices)


def _evaluate_trade_limits(
    policy: RiskPolicy,
    trade: Trade,
    notional: Decimal,
    nav: Decimal,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
) -> PolicyResult:
    """Evaluate per-trade and name-concentration limits; return a PolicyResult."""
    trade_pct = notional / nav

    if trade_pct > policy.max_trade_notional_pct:
        return Rejected(
            reason=(
                f"Trade notional {trade_pct:.2%} exceeds max "
                f"{policy.max_trade_notional_pct:.2%} of NAV."
            )
        )

    if trade_pct > policy.hitl_threshold_pct:
        return HITLRequired(
            reason=(
                f"Trade notional {trade_pct:.2%} exceeds HITL "
                f"threshold {policy.hitl_threshold_pct:.2%} of NAV."
            )
        )

    if trade.side == "buy":
        projected = _projected_weight(trade, portfolio, prices, nav)
        if projected > policy.max_name_concentration_pct:
            return Rejected(
                reason=(
                    f"Post-trade weight {projected:.2%} exceeds max "
                    f"name concentration {policy.max_name_concentration_pct:.2%}."
                )
            )

    return Approved()


def _projected_weight(
    trade: Trade,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    nav: Decimal,
) -> Decimal:
    """Projected single-name weight after a buy, assuming constant NAV."""
    current_qty = Decimal("0")
    if trade.symbol in portfolio.holdings:
        current_qty = portfolio.holdings[trade.symbol].quantity
    new_qty = current_qty + trade.qty
    return new_qty * prices.get(trade.symbol, trade.requested_price) / nav


# ---------------------------------------------------------------------------
# Trade — lifecycle state machine
# ---------------------------------------------------------------------------


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
        """Re-check limits against current bar before execution.

        The requested price is updated to the bar's close so limits are
        evaluated against the current market, not the stale approval price.
        Pass ``start_of_day_nav`` to enforce the daily-loss halt during
        revalidation.
        """
        current_trade = self.model_copy(update={"requested_price": bar.close})
        return risk.check_trade(current_trade, portfolio, prices, start_of_day_nav)


# ---------------------------------------------------------------------------
# DecisionCycle — one complete research → PM → risk → execution cycle
# ---------------------------------------------------------------------------


class DecisionCycle(BaseModel):
    """One end-to-end decision cycle, keyed by correlation_id.

    ``id`` doubles as the ``correlation_id`` that propagates through every
    agent invocation, tool call, and trade within this cycle.
    """

    id: UUID = Field(default_factory=uuid4)
    trigger_type: Literal["scheduled", "event"]
    trigger_ref: str | None = None  # e.g. headline hash for event triggers
    started_at: datetime
    outcome: str | None = None  # e.g. "filled", "rejected", "refusal", "hitl_expired"

    model_config = {"frozen": False}
