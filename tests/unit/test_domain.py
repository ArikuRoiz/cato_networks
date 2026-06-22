"""Unit tests for src/firm/domain/entities.py.

All tests are pure-Python — no IO, no DB, no network.
Coverage target: ≥ 90% on src/firm/domain/.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from firm.domain.entities import (
    Approved,
    Bar,
    HITLRequired,
    Holding,
    InsufficientCash,
    InsufficientHolding,
    Lot,
    Portfolio,
    Rejected,
    RiskPolicy,
    Trade,
    TradeStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime(2024, 10, 21, 9, 30, 0, tzinfo=UTC)


def _lot(symbol: str, qty: str, cost: str, offset_seconds: int = 0) -> Lot:
    base = datetime(2024, 10, 21, 9, 30, 0, tzinfo=UTC)
    ts = base + timedelta(seconds=offset_seconds)
    return Lot(symbol=symbol, qty=Decimal(qty), cost=Decimal(cost), opened_at=ts)


def _trade(
    symbol: str,
    side: str,
    qty: str,
    price: str,
    cycle_id=None,
) -> Trade:
    return Trade(
        cycle_id=cycle_id or uuid4(),
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=Decimal(qty),
        requested_price=Decimal(price),
        idempotency_key=f"{symbol}-{side}-{qty}-{price}",
    )


def _policy(
    max_trade: str = "0.10",
    max_conc: str = "0.25",
    halt: str = "0.03",
    hitl: str = "0.05",
) -> RiskPolicy:
    return RiskPolicy(
        max_trade_notional_pct=Decimal(max_trade),
        max_name_concentration_pct=Decimal(max_conc),
        daily_loss_halt_pct=Decimal(halt),
        hitl_threshold_pct=Decimal(hitl),
    )


def _bar(symbol: str, close: str) -> Bar:
    p = Decimal(close)
    return Bar(
        symbol=symbol,
        open=p,
        high=p,
        low=p,
        close=p,
        volume=1_000_000,
        ts=_utcnow(),
    )


# ---------------------------------------------------------------------------
# Lot / Holding FIFO tests
# ---------------------------------------------------------------------------


def test_fifo_lot_partial_close() -> None:
    """Buy 10 @ $100, buy 5 @ $110 — close 7 should consume only the first lot."""
    lot1 = _lot("AAPL", "10", "100")
    lot2 = _lot("AAPL", "5", "110", offset_seconds=60)
    holding = Holding(symbol="AAPL", lots=[lot1, lot2])

    pairs = holding.close_lots(Decimal("7"))

    assert len(pairs) == 1
    consumed_lot, consumed_qty = pairs[0]
    assert consumed_lot.id == lot1.id
    assert consumed_qty == Decimal("7")


def test_fifo_lot_multi_lot() -> None:
    """Close more than first lot — spans two lots in FIFO order."""
    lot1 = _lot("AAPL", "10", "100")
    lot2 = _lot("AAPL", "5", "110", offset_seconds=60)
    holding = Holding(symbol="AAPL", lots=[lot1, lot2])

    pairs = holding.close_lots(Decimal("12"))

    assert len(pairs) == 2
    first_lot, first_qty = pairs[0]
    second_lot, second_qty = pairs[1]
    assert first_lot.id == lot1.id
    assert first_qty == Decimal("10")
    assert second_lot.id == lot2.id
    assert second_qty == Decimal("2")


def test_fifo_lot_out_of_order_insertion() -> None:
    """Lots inserted newest-first are still consumed oldest-first (FIFO by opened_at)."""
    lot_old = _lot("AAPL", "10", "100", offset_seconds=0)
    lot_new = _lot("AAPL", "5", "110", offset_seconds=60)
    # Deliberately insert newer lot first to expose sort dependency
    holding = Holding(symbol="AAPL", lots=[lot_new, lot_old])

    pairs = holding.close_lots(Decimal("7"))

    assert len(pairs) == 1
    consumed_lot, consumed_qty = pairs[0]
    assert consumed_lot.id == lot_old.id  # oldest consumed first
    assert consumed_qty == Decimal("7")


def test_close_lots_raises_insufficient_holding() -> None:
    """close_lots raises InsufficientHolding when qty > open position."""
    lot = _lot("NVDA", "5", "120")
    holding = Holding(symbol="NVDA", lots=[lot])

    with pytest.raises(InsufficientHolding):
        holding.close_lots(Decimal("6"))


def test_holding_quantity_and_avg_cost() -> None:
    """Quantity sums all lots; avg_cost is share-weighted."""
    lot1 = _lot("MSFT", "10", "100")
    lot2 = _lot("MSFT", "10", "110")
    holding = Holding(symbol="MSFT", lots=[lot1, lot2])

    assert holding.quantity == Decimal("20")
    assert holding.avg_cost == Decimal("105")


def test_holding_open_lots_filters_zero_qty() -> None:
    """open_lots() returns only lots with qty > 0."""
    lot1 = _lot("AMD", "0", "80")
    lot2 = _lot("AMD", "5", "85")
    holding = Holding(symbol="AMD", lots=[lot1, lot2])

    open_lots = holding.open_lots()

    assert len(open_lots) == 1
    assert open_lots[0].id == lot2.id


# ---------------------------------------------------------------------------
# Portfolio affordability tests
# ---------------------------------------------------------------------------


def test_can_afford_boundary_true() -> None:
    """Cash exactly covers notional + commission — can_afford returns True."""
    qty = Decimal("100")
    price = Decimal("10")
    commission = qty * Decimal("0.005")  # $0.50
    cash = qty * price + commission  # exactly covers

    portfolio = Portfolio(id=uuid4(), cash=cash)

    assert portfolio.can_afford("AAPL", qty, price) is True


def test_can_afford_boundary_false() -> None:
    """One cent short — can_afford raises InsufficientCash."""
    qty = Decimal("100")
    price = Decimal("10")
    commission = qty * Decimal("0.005")
    cash = qty * price + commission - Decimal("0.01")

    portfolio = Portfolio(id=uuid4(), cash=cash)

    with pytest.raises(InsufficientCash):
        portfolio.can_afford("AAPL", qty, price)


def test_insufficient_cash_message() -> None:
    """InsufficientCash carries a useful message identifying symbol and amounts."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("0"))

    with pytest.raises(InsufficientCash, match="AAPL"):
        portfolio.can_afford("AAPL", Decimal("1"), Decimal("100"))


# ---------------------------------------------------------------------------
# Portfolio NAV tests
# ---------------------------------------------------------------------------


def test_nav_calculation() -> None:
    """NAV = cash + sum(qty * price) for each holding."""
    aapl_lot = _lot("AAPL", "10", "150")
    msft_lot = _lot("MSFT", "5", "300")
    portfolio = Portfolio(
        id=uuid4(),
        cash=Decimal("5000"),
        holdings={
            "AAPL": Holding(symbol="AAPL", lots=[aapl_lot]),
            "MSFT": Holding(symbol="MSFT", lots=[msft_lot]),
        },
    )
    prices = {"AAPL": Decimal("200"), "MSFT": Decimal("400")}

    # expected: 5000 + 10*200 + 5*400 = 5000 + 2000 + 2000 = 9000
    assert portfolio.nav(prices) == Decimal("9000")


def test_nav_no_holdings() -> None:
    """NAV of a cash-only portfolio equals cash."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("10000"))
    assert portfolio.nav({}) == Decimal("10000")


def test_position_weight() -> None:
    """Position weight is holding_value / nav."""
    aapl_lot = _lot("AAPL", "10", "100")
    portfolio = Portfolio(
        id=uuid4(),
        cash=Decimal("8000"),
        holdings={"AAPL": Holding(symbol="AAPL", lots=[aapl_lot])},
    )
    prices = {"AAPL": Decimal("200")}
    # nav = 8000 + 10*200 = 10000; weight = 2000/10000 = 0.2
    assert portfolio.position_weight("AAPL", prices) == Decimal("0.2")


# ---------------------------------------------------------------------------
# RiskPolicy tests
# ---------------------------------------------------------------------------


def test_risk_policy_approved() -> None:
    """Small trade well below all limits → Approved."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("100000"))
    prices: dict[str, Decimal] = {}
    policy = _policy(max_trade="0.10", hitl="0.05")

    # notional = 1 * 100 = 100; nav = 100000; pct = 0.001 → well under 5 %
    trade = _trade("AAPL", "buy", "1", "100")
    result = policy.check_trade(trade, portfolio, prices)

    assert isinstance(result, Approved)


def test_risk_policy_hitl_required() -> None:
    """Trade notional > hitl_threshold but ≤ max_trade → HITLRequired."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("100000"))
    prices: dict[str, Decimal] = {}
    policy = _policy(max_trade="0.10", hitl="0.05")

    # notional = 700 * 10 = 7000; nav ≈ 100000; pct = 0.07 → above 5%, below 10%
    trade = _trade("NVDA", "buy", "700", "10")
    result = policy.check_trade(trade, portfolio, prices)

    assert isinstance(result, HITLRequired)


def test_risk_policy_rejected() -> None:
    """Trade notional > max_trade_notional_pct → Rejected."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("100000"))
    prices: dict[str, Decimal] = {}
    policy = _policy(max_trade="0.10", hitl="0.05")

    # notional = 1200 * 10 = 12000; nav = 100000; pct = 0.12 → above 10%
    trade = _trade("NVDA", "buy", "1200", "10")
    result = policy.check_trade(trade, portfolio, prices)

    assert isinstance(result, Rejected)


def test_risk_policy_concentration_rejected() -> None:
    """Buy that would push single-name weight above max_name_concentration_pct → Rejected."""
    aapl_lot = _lot("AAPL", "200", "100")
    portfolio = Portfolio(
        id=uuid4(),
        cash=Decimal("60000"),
        holdings={"AAPL": Holding(symbol="AAPL", lots=[aapl_lot])},
    )
    # nav = 60000 + 200*100 = 80000; existing weight = 20000/80000 = 25%
    prices = {"AAPL": Decimal("100")}
    # Adding even 1 share would exceed 25% concentration
    policy = _policy(max_trade="0.10", max_conc="0.25", hitl="0.05")
    trade = _trade("AAPL", "buy", "1", "100")
    result = policy.check_trade(trade, portfolio, prices)

    assert isinstance(result, Rejected)


def test_risk_policy_zero_nav_rejected() -> None:
    """Portfolio with zero NAV → Rejected (cannot size trade)."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("0"))
    policy = _policy()
    trade = _trade("AAPL", "buy", "1", "100")
    result = policy.check_trade(trade, portfolio, {})

    assert isinstance(result, Rejected)


# ---------------------------------------------------------------------------
# Daily-loss halt tests (LOCKED DECISION: -3 % daily NAV)
# ---------------------------------------------------------------------------


def test_daily_halt_not_triggered() -> None:
    """Loss below threshold — check_daily_halt returns Approved."""
    policy = _policy(halt="0.03")
    start_nav = Decimal("100000")
    current_nav = Decimal("98000")  # -2 % loss, below 3 % threshold

    result = policy.check_daily_halt(current_nav, start_nav)

    assert isinstance(result, Approved)


def test_daily_halt_triggered_at_threshold() -> None:
    """Loss exactly at threshold — check_daily_halt returns Rejected."""
    policy = _policy(halt="0.03")
    start_nav = Decimal("100000")
    current_nav = Decimal("97000")  # exactly -3 %

    result = policy.check_daily_halt(current_nav, start_nav)

    assert isinstance(result, Rejected)
    assert "3.00%" in result.reason


def test_daily_halt_triggered_above_threshold() -> None:
    """Loss exceeds threshold — check_daily_halt returns Rejected."""
    policy = _policy(halt="0.03")
    start_nav = Decimal("100000")
    current_nav = Decimal("96000")  # -4 % loss

    result = policy.check_daily_halt(current_nav, start_nav)

    assert isinstance(result, Rejected)


def test_check_trade_halted_when_daily_loss_exceeded() -> None:
    """check_trade with start_of_day_nav that has exceeded halt → Rejected."""
    # Portfolio has lost 4 % intraday — breach the -3 % halt
    portfolio = Portfolio(id=uuid4(), cash=Decimal("96000"))
    policy = _policy(halt="0.03")
    trade = _trade("AAPL", "buy", "1", "100")
    start_nav = Decimal("100000")

    result = policy.check_trade(trade, portfolio, {}, start_of_day_nav=start_nav)

    assert isinstance(result, Rejected)
    assert "halted" in result.reason.lower()


def test_check_trade_skips_halt_when_start_nav_omitted() -> None:
    """check_trade without start_of_day_nav skips halt — small trade is Approved."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("96000"))
    policy = _policy(halt="0.03")
    trade = _trade("AAPL", "buy", "1", "100")

    result = policy.check_trade(trade, portfolio, {})

    assert isinstance(result, Approved)


# ---------------------------------------------------------------------------
# Trade.revalidate tests
# ---------------------------------------------------------------------------


def test_trade_revalidate_passes_with_current_bar() -> None:
    """revalidate with a small trade and current bar → Approved."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("100000"))
    policy = _policy()
    trade = _trade("AAPL", "buy", "1", "100")
    bar = _bar("AAPL", "100")
    prices: dict[str, Decimal] = {}

    result = trade.revalidate(bar, policy, portfolio, prices)

    assert isinstance(result, Approved)


def test_trade_revalidate_blocked_after_price_move() -> None:
    """Revalidate using current bar close — if price moved high enough, trade is rejected."""
    portfolio = Portfolio(id=uuid4(), cash=Decimal("100000"))
    policy = _policy(max_trade="0.10", hitl="0.05")
    # Original price approved; bar close is now very high → triggers rejection
    trade = _trade("NVDA", "buy", "1200", "1")
    bar = _bar("NVDA", "10")  # bar.close = 10 → notional = 12000, pct = 12%
    prices: dict[str, Decimal] = {}

    result = trade.revalidate(bar, policy, portfolio, prices)

    assert isinstance(result, Rejected)


# ---------------------------------------------------------------------------
# TradeStatus enum
# ---------------------------------------------------------------------------


def test_trade_status_values() -> None:
    """All expected TradeStatus members are present."""
    expected = {
        "PROPOSED",
        "PENDING_HITL",
        "APPROVED",
        "REJECTED",
        "FILLED",
        "FAILED",
        "EXPIRED",
    }
    actual = {s.value for s in TradeStatus}
    assert actual == expected


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------


def test_bar_immutable() -> None:
    """Bar is frozen — mutation raises ValidationError."""
    bar = _bar("SPY", "500")
    with pytest.raises((TypeError, ValidationError)):
        bar.close = Decimal("999")  # type: ignore[misc]
