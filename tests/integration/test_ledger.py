"""Integration tests for LedgerRepository against an ephemeral Postgres instance.

Uses testcontainers to spin up Postgres, runs the Alembic migrations, then exercises
the three acceptance-criteria scenarios from FIRM-4:

  test_crash_mid_trade_reconciles  — ACID: mid-transaction failure leaves no partial state
  test_idempotent_execution        — duplicate idempotency_key is a no-op
  test_ledger_fifo_sell            — FIFO lot math: buy 10@100 + 5@110 → sell 7 → correct lots

These tests remove the ``xfail`` stubs from tests/integration/test_mandatory.py.

Requires: testcontainers[postgres], alembic, sqlalchemy, psycopg (binary).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from firm.domain.entities import Trade, TradeStatus
from firm.persistence.ledger import LedgerRepository

# ---------------------------------------------------------------------------
# Project-root anchor (works regardless of caller cwd)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Session-scoped Postgres container + migrated engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_container():
    """Start a Postgres 16 container for the entire test session."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def migrated_engine(pg_container: PostgresContainer) -> Engine:
    """Create an engine pointing at the test container and run all migrations."""
    url = pg_container.get_connection_url()
    # testcontainers returns a psycopg2-style URL; normalise to psycopg3
    url = url.replace("psycopg2", "psycopg").replace(
        "postgresql+psycopg://", "postgresql+psycopg://"
    )
    engine = create_engine(url, echo=False)
    _run_migrations(engine, url)
    return engine


def _run_migrations(engine: Engine, url: str) -> None:
    """Run Alembic migrations against *engine* using DATABASE_URL override."""
    ini_path = _PROJECT_ROOT / "migrations" / "alembic.ini"
    cfg = AlembicConfig(str(ini_path))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    alembic_command.upgrade(cfg, "head")


# ---------------------------------------------------------------------------
# Per-test portfolio factory
# ---------------------------------------------------------------------------


def _make_portfolio(engine: Engine, cash: Decimal) -> uuid.UUID:
    """Insert a fresh portfolio row and return its id."""
    portfolio_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO portfolios (id, cash_balance, created_at) VALUES (:id, :cash, :ts)"),
            {"id": str(portfolio_id), "cash": str(cash), "ts": datetime.now(tz=UTC)},
        )
    return portfolio_id


def _build_trade(
    portfolio_id: uuid.UUID,
    *,
    symbol: str = "AAPL",
    side: str = "buy",
    qty: Decimal = Decimal("10"),
    price: Decimal = Decimal("100.00"),
    key: str | None = None,
) -> Trade:
    cycle_id = uuid.uuid4()
    return Trade(
        id=uuid.uuid4(),
        cycle_id=cycle_id,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        requested_price=price,
        idempotency_key=key or uuid.uuid4().hex,
    )


# ---------------------------------------------------------------------------
# test_crash_mid_trade_reconciles
# ---------------------------------------------------------------------------


def test_crash_mid_trade_reconciles(migrated_engine: Engine) -> None:
    """Simulating a crash mid-transaction (LotRow insert raises) must leave the
    portfolio in its original state — cash not debited, no trade row inserted.

    This turns the xfail stub in test_mandatory.py green for:
      FIRM-4 / FR-1 crash recovery requirement.
    """
    initial_cash = Decimal("50000.00")
    portfolio_id = _make_portfolio(migrated_engine, initial_cash)
    repo = LedgerRepository(migrated_engine)

    trade = _build_trade(portfolio_id, qty=Decimal("10"), price=Decimal("100.00"))

    # Patch _insert_lot to raise mid-transaction
    with pytest.raises(RuntimeError, match="simulated crash"):
        with patch(
            "firm.persistence.ledger._insert_lot",
            side_effect=RuntimeError("simulated crash"),
        ):
            repo.buy(trade, portfolio_id)

    # Portfolio must be unchanged — cash not debited
    portfolio = repo.get_portfolio(portfolio_id)
    assert portfolio.cash == initial_cash, (
        f"Cash should be {initial_cash} after rolled-back buy, got {portfolio.cash}"
    )

    # No trade row should exist for this trade id
    fetched = repo.get_trade(trade.id)
    assert fetched is None, "TradeRow must not exist after a rolled-back transaction."


# ---------------------------------------------------------------------------
# test_idempotent_execution
# ---------------------------------------------------------------------------


def test_idempotent_execution(migrated_engine: Engine) -> None:
    """Calling buy() twice with the same idempotency_key must:
      - Insert exactly one TradeRow
      - Debit cash exactly once
      - Return the same Trade both calls

    This turns the xfail stub in test_mandatory.py green for:
      FIRM-4 / idempotency_key unique constraint requirement.
    """
    initial_cash = Decimal("50000.00")
    portfolio_id = _make_portfolio(migrated_engine, initial_cash)
    repo = LedgerRepository(migrated_engine)

    shared_key = "idempotency-test-key-001"
    trade = _build_trade(
        portfolio_id,
        qty=Decimal("5"),
        price=Decimal("200.00"),
        key=shared_key,
    )

    first = repo.buy(trade, portfolio_id)
    second = repo.buy(trade, portfolio_id)

    assert first.id == second.id, "Both calls must return the same trade."
    assert first.status == TradeStatus.FILLED
    assert second.status == TradeStatus.FILLED

    # Exactly one TradeRow must exist for this idempotency_key — not two.
    with migrated_engine.connect() as conn:
        row_count = conn.execute(
            text("SELECT COUNT(*) FROM trades WHERE idempotency_key = :key"),
            {"key": shared_key},
        ).scalar_one()
    assert row_count == 1, (
        f"Expected exactly 1 TradeRow for idempotency_key '{shared_key}', found {row_count}."
    )

    # Cash debited exactly once
    portfolio = repo.get_portfolio(portfolio_id)
    notional = trade.qty * trade.requested_price
    commission = trade.qty * Decimal("0.005")
    # Slippage (5 bps) is also applied on buy
    # Cash must be below initial (debited once) but above (initial - 2*notional)
    assert portfolio.cash < initial_cash, "Cash must have been debited at least once."
    assert portfolio.cash > initial_cash - 2 * (notional + commission), (
        "Cash must have been debited only once, not twice."
    )


# ---------------------------------------------------------------------------
# test_ledger_fifo_sell
# ---------------------------------------------------------------------------


def test_ledger_fifo_sell(migrated_engine: Engine) -> None:
    """Buy 10 @ $100 then 5 @ $110, sell 7 → FIFO closes the $100 lot first.

    After sell:
      - Lot 1 (10 shares @ $100) should have 3 shares remaining.
      - Lot 2 (5 shares @ $110) should still have 5 shares.
      - Cash should reflect: proceeds from sell minus commission.
    """
    initial_cash = Decimal("100000.00")
    portfolio_id = _make_portfolio(migrated_engine, initial_cash)
    repo = LedgerRepository(migrated_engine)

    # Anchor time: guarantee a 1-second gap between lot opened_at values so
    # that FIFO sort by opened_at is deterministic even on fast CI machines.
    _t0 = datetime(2024, 10, 21, 9, 30, 0, tzinfo=UTC)
    _t1 = _t0 + timedelta(seconds=1)

    # First buy: 10 @ $100 — lot opened_at = _t0 (injected explicitly for deterministic FIFO)
    buy1 = _build_trade(portfolio_id, qty=Decimal("10"), price=Decimal("100.00"))
    repo.buy(buy1, portfolio_id, opened_at=_t0)

    # Second buy: 5 @ $110 — lot opened_at = _t1 (one second later, guaranteed FIFO order)
    buy2 = _build_trade(portfolio_id, qty=Decimal("5"), price=Decimal("110.00"))
    repo.buy(buy2, portfolio_id, opened_at=_t1)

    # Confirm position before sell
    portfolio_after_buys = repo.get_portfolio(portfolio_id)
    assert "AAPL" in portfolio_after_buys.holdings
    assert portfolio_after_buys.holdings["AAPL"].quantity == Decimal("15")

    # Sell 7 — FIFO closes from the $100 lot first
    sell_trade = _build_trade(
        portfolio_id,
        side="sell",
        qty=Decimal("7"),
        price=Decimal("105.00"),
    )
    filled_sell = repo.sell(sell_trade, portfolio_id)

    assert filled_sell.status == TradeStatus.FILLED

    # Check remaining lots via domain model
    portfolio_after_sell = repo.get_portfolio(portfolio_id)
    holding = portfolio_after_sell.holdings["AAPL"]
    assert holding.quantity == Decimal("8"), (
        f"Expected 8 remaining (10-7+5), got {holding.quantity}"
    )

    # Verify FIFO: the first lot should have 3 remaining, second lot intact at 5
    open_lots = sorted(holding.open_lots(), key=lambda lot: lot.opened_at)
    assert len(open_lots) == 2, f"Expected 2 lots, got {len(open_lots)}"
    first_lot_qty = open_lots[0].qty
    second_lot_qty = open_lots[1].qty
    assert first_lot_qty == Decimal("3"), (
        f"First lot (cost=$100) should have 3 remaining after FIFO sell of 7, got {first_lot_qty}"
    )
    assert second_lot_qty == Decimal("5"), (
        f"Second lot (cost=$110) should be untouched, got {second_lot_qty}"
    )

    # Cash: sell proceeds credited (qty * fill_price - commission)
    commission = Decimal("7") * Decimal("0.005")
    # fill_price = requested * (1 - slippage_bps) for a sell
    sell_fill_price = Decimal("105.00") * (Decimal("1") - Decimal("0.0005"))
    expected_proceeds = Decimal("7") * sell_fill_price - commission
    assert portfolio_after_sell.cash > portfolio_after_buys.cash, "Cash must increase after a sell."
    # Tolerance: allow for minor Decimal arithmetic differences
    cash_gained = portfolio_after_sell.cash - portfolio_after_buys.cash
    assert abs(cash_gained - expected_proceeds) < Decimal("0.01"), (
        f"Cash gained {cash_gained} does not match expected proceeds {expected_proceeds}."
    )
