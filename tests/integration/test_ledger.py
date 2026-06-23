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

from firm.domain import Trade, TradeStatus
from firm.persistence.ledger import CycleAuditRecord, LedgerRepository
from firm.persistence.models import ApprovalRow, AuditLogRow, DecisionCycleRow

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
    """Run Alembic migrations against *engine* using DATABASE_URL override.

    Sets DATABASE_URL in the environment so migrations/env.py picks it up
    (it reads DATABASE_URL first, which would otherwise override the URL we
    set via set_main_option when running in CI where DATABASE_URL is set).
    """
    import os

    ini_path = _PROJECT_ROOT / "migrations" / "alembic.ini"
    cfg = AlembicConfig(str(ini_path))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        alembic_command.upgrade(cfg, "head")
    finally:
        if old_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_url


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


# ---------------------------------------------------------------------------
# test_record_approval_writes_row_and_audit_entry (R6)
# ---------------------------------------------------------------------------


def test_record_approval_writes_row_and_audit_entry(migrated_engine: Engine) -> None:
    """record_approval must write one ApprovalRow and one AuditLogRow atomically.

    Steps:
      1. Create a portfolio and buy a trade so a valid TradeRow FK exists.
      2. Call record_approval for each HITL status variant.
      3. Assert ApprovalRow fields match what was passed in.
      4. Assert AuditLogRow has action='hitl.decision' and full payload.
    """
    from sqlalchemy.orm import Session

    initial_cash = Decimal("50000.00")
    portfolio_id = _make_portfolio(migrated_engine, initial_cash)
    repo = LedgerRepository(migrated_engine)

    trade = _build_trade(portfolio_id, qty=Decimal("10"), price=Decimal("100.00"))
    filled = repo.buy(trade, portfolio_id)
    assert filled.id is not None

    correlation_id = uuid.uuid4()
    original_notional = Decimal("1005.05")
    original_qty = Decimal("10")

    repo.record_approval(
        correlation_id=correlation_id,
        trade_id=filled.id,
        status="approved",
        original_notional=original_notional,
        original_qty=original_qty,
        decided_by="risk_committee",
    )

    with Session(migrated_engine) as session:
        approval_rows = session.query(ApprovalRow).filter_by(trade_id=filled.id).all()
        assert len(approval_rows) == 1, (
            f"Expected 1 ApprovalRow for trade {filled.id}, got {len(approval_rows)}"
        )
        row = approval_rows[0]
        assert row.status == "approved"
        assert row.decided_by == "risk_committee"
        assert row.decided_at is not None

        audit_rows = (
            session.query(AuditLogRow)
            .filter_by(correlation_id=correlation_id, action="hitl.decision")
            .all()
        )
        assert len(audit_rows) == 1, (
            f"Expected 1 AuditLogRow with action='hitl.decision', got {len(audit_rows)}"
        )
        payload = audit_rows[0].payload
        assert payload["status"] == "approved"
        assert payload["trade_id"] == str(filled.id)
        assert payload["correlation_id"] == str(correlation_id)
        assert payload["original_notional"] == str(original_notional)
        assert payload["original_qty"] == str(original_qty)
        assert "edited_qty" not in payload


def test_record_approval_includes_edited_qty_when_provided(migrated_engine: Engine) -> None:
    """record_approval with edited_qty must include it in the audit payload."""
    from sqlalchemy.orm import Session

    portfolio_id = _make_portfolio(migrated_engine, Decimal("50000.00"))
    repo = LedgerRepository(migrated_engine)

    trade = _build_trade(portfolio_id, qty=Decimal("5"), price=Decimal("200.00"))
    filled = repo.buy(trade, portfolio_id)

    correlation_id = uuid.uuid4()
    repo.record_approval(
        correlation_id=correlation_id,
        trade_id=filled.id,
        status="edited",
        original_notional=Decimal("1000"),
        original_qty=Decimal("5"),
        edited_qty=Decimal("3"),
    )

    with Session(migrated_engine) as session:
        audit_rows = (
            session.query(AuditLogRow)
            .filter_by(correlation_id=correlation_id, action="hitl.decision")
            .all()
        )
        assert len(audit_rows) == 1
        payload = audit_rows[0].payload
        assert payload["edited_qty"] == "3"
        assert payload["status"] == "edited"


# ---------------------------------------------------------------------------
# test_record_cycle_hold_writes_decision_cycle_and_audit_entries
# ---------------------------------------------------------------------------


def test_record_cycle_hold_writes_decision_cycle_and_audit_entries(
    migrated_engine: Engine,
) -> None:
    """A Hold cycle must write one decision_cycles row and four audit_log entries.

    This is the key auditability invariant: every cycle leaves a trace, regardless
    of outcome.  Hold cycles previously wrote zero rows (the bug this fixes).

    Steps:
      1. Build a CycleAuditRecord with outcome="hold" and no trade_id.
      2. Call record_cycle on a LedgerRepository.
      3. Assert one DecisionCycleRow exists with correct trigger_ref and outcome.
      4. Assert exactly four AuditLogRow entries exist for the correlation_id:
           research.done, decision.made, risk.outcome, cycle.outcome.
      5. Assert cycle.outcome payload carries the symbol and outcome; no trade_id key.
    """
    from sqlalchemy.orm import Session

    correlation_id = str(uuid.uuid4())
    decision_ts = datetime(2024, 10, 21, 10, 0, 0, tzinfo=UTC)
    repo = LedgerRepository(migrated_engine)

    record = CycleAuditRecord(
        correlation_id=correlation_id,
        symbol="NVDA",
        trigger_type="scheduled",
        decision_ts=decision_ts,
        recommendation="hold",
        conviction=0.45,
        outcome="hold",
        judge_score=3,
        alignment="partial",
        trade_id=None,
    )
    repo.record_cycle(record)

    with Session(migrated_engine) as session:
        # One decision_cycles row
        cycle_rows = session.query(DecisionCycleRow).filter_by(trigger_ref=correlation_id).all()
        assert len(cycle_rows) == 1, (
            f"Expected 1 DecisionCycleRow for correlation_id={correlation_id}, "
            f"got {len(cycle_rows)}"
        )
        cycle_row = cycle_rows[0]
        assert cycle_row.trigger_type == "scheduled"
        assert cycle_row.outcome == "hold"

        # Four audit_log entries
        correlation_uuid = uuid.UUID(correlation_id)
        audit_rows = session.query(AuditLogRow).filter_by(correlation_id=correlation_uuid).all()
        actions = {row.action for row in audit_rows}
        expected_actions = {"research.done", "decision.made", "risk.outcome", "cycle.outcome"}
        assert actions == expected_actions, (
            f"Expected audit actions {expected_actions}, got {actions}"
        )

        # cycle.outcome payload has key fields and no trade_id (this is a hold)
        outcome_rows = [r for r in audit_rows if r.action == "cycle.outcome"]
        assert len(outcome_rows) == 1
        payload = outcome_rows[0].payload
        assert payload["symbol"] == "NVDA"
        assert payload["outcome"] == "hold"
        assert payload["judge_score"] == 3
        assert payload["alignment"] == "partial"
        assert "trade_id" not in payload, "Hold cycle must not have trade_id in payload"


def test_record_cycle_filled_writes_trade_id_in_payload(migrated_engine: Engine) -> None:
    """A filled cycle must include trade_id in the cycle.outcome audit payload.

    The correlation_id ↔ trade_id link is what makes `make trace TRADE=<id>` work.
    """
    from sqlalchemy.orm import Session

    correlation_id = str(uuid.uuid4())
    trade_id = uuid.uuid4()
    decision_ts = datetime(2024, 10, 22, 14, 30, 0, tzinfo=UTC)
    repo = LedgerRepository(migrated_engine)

    record = CycleAuditRecord(
        correlation_id=correlation_id,
        symbol="AAPL",
        trigger_type="event",
        decision_ts=decision_ts,
        recommendation="strong_buy",
        conviction=0.85,
        outcome="filled",
        judge_score=5,
        alignment="aligned",
        trade_id=trade_id,
    )
    repo.record_cycle(record)

    with Session(migrated_engine) as session:
        cycle_rows = session.query(DecisionCycleRow).filter_by(trigger_ref=correlation_id).all()
        assert len(cycle_rows) == 1
        assert cycle_rows[0].outcome == "filled"

        correlation_uuid = uuid.UUID(correlation_id)
        outcome_rows = (
            session.query(AuditLogRow)
            .filter_by(correlation_id=correlation_uuid, action="cycle.outcome")
            .all()
        )
        assert len(outcome_rows) == 1
        payload = outcome_rows[0].payload
        assert payload["trade_id"] == str(trade_id), (
            "Filled cycle must link trade_id in cycle.outcome payload for trace queries"
        )


# ---------------------------------------------------------------------------
# ensure_portfolio — idempotent portfolio seeding
# ---------------------------------------------------------------------------


def test_ensure_portfolio_creates_row(migrated_engine: Engine) -> None:
    """ensure_portfolio must write a portfolio row with the given cash balance.

    A second call with the same portfolio_id must be a no-op (ON CONFLICT DO NOTHING).
    """
    from sqlalchemy.orm import Session

    from firm.persistence.models import PortfolioRow

    repo = LedgerRepository(migrated_engine)
    portfolio_id = uuid.uuid4()
    starting_cash = Decimal("100000.00")

    # First call: row is created.
    repo.ensure_portfolio(portfolio_id, starting_cash)

    with Session(migrated_engine) as session:
        row = session.get(PortfolioRow, portfolio_id)
        assert row is not None, "ensure_portfolio must create a PortfolioRow"
        assert row.cash_balance == starting_cash

    # Second call: no exception, no duplicate row.
    repo.ensure_portfolio(portfolio_id, starting_cash)
    with Session(migrated_engine) as session:
        rows = session.query(PortfolioRow).filter_by(id=portfolio_id).all()
        assert len(rows) == 1, "ensure_portfolio must be idempotent — no duplicate row"


def test_ensure_portfolio_does_not_overwrite_cash(migrated_engine: Engine) -> None:
    """ensure_portfolio must not reduce cash when the row already exists."""
    from sqlalchemy.orm import Session

    from firm.persistence.models import PortfolioRow

    repo = LedgerRepository(migrated_engine)
    portfolio_id = _make_portfolio(migrated_engine, Decimal("200000.00"))

    # Calling with a lower cash value must leave the original row untouched.
    repo.ensure_portfolio(portfolio_id, Decimal("50000.00"))

    with Session(migrated_engine) as session:
        row = session.get(PortfolioRow, portfolio_id)
        assert row is not None
        assert row.cash_balance == Decimal("200000.00"), (
            "ensure_portfolio must not overwrite existing cash_balance"
        )


# ---------------------------------------------------------------------------
# Pending-run registry
# ---------------------------------------------------------------------------


def test_register_and_list_pending_runs(migrated_engine: Engine) -> None:
    """register_pending_run must write a row; list_pending_runs must return it."""
    repo = LedgerRepository(migrated_engine)
    thread_id = f"thread-{uuid.uuid4().hex}"
    correlation_id = str(uuid.uuid4())
    symbol = "NVDA"

    repo.register_pending_run(
        thread_id=thread_id,
        correlation_id=correlation_id,
        symbol=symbol,
    )

    runs = repo.list_pending_runs()
    matching = [(t, c, s) for t, c, s in runs if t == thread_id]
    assert len(matching) == 1, f"Expected 1 pending run for {thread_id}, got {matching}"
    assert matching[0][1] == correlation_id
    assert matching[0][2] == symbol


def test_register_pending_run_idempotent(migrated_engine: Engine) -> None:
    """Registering the same thread_id twice must not raise — ON CONFLICT DO NOTHING."""
    repo = LedgerRepository(migrated_engine)
    thread_id = f"thread-{uuid.uuid4().hex}"

    repo.register_pending_run(
        thread_id=thread_id, correlation_id=str(uuid.uuid4()), symbol="AAPL"
    )
    repo.register_pending_run(
        thread_id=thread_id, correlation_id=str(uuid.uuid4()), symbol="AAPL"
    )

    runs = repo.list_pending_runs()
    matching = [t for t, _, _ in runs if t == thread_id]
    assert len(matching) == 1, "Duplicate register must produce exactly one row"


def test_delete_pending_run_removes_row(migrated_engine: Engine) -> None:
    """delete_pending_run must remove the row; subsequent list must not include it."""
    repo = LedgerRepository(migrated_engine)
    thread_id = f"thread-{uuid.uuid4().hex}"

    repo.register_pending_run(
        thread_id=thread_id, correlation_id=str(uuid.uuid4()), symbol="MSFT"
    )
    repo.delete_pending_run(thread_id)

    runs = repo.list_pending_runs()
    matching = [t for t, _, _ in runs if t == thread_id]
    assert matching == [], f"delete_pending_run must remove the row; still found {matching}"


def test_delete_nonexistent_pending_run_is_noop(migrated_engine: Engine) -> None:
    """Deleting a thread_id that does not exist must not raise."""
    repo = LedgerRepository(migrated_engine)
    # Must not raise
    repo.delete_pending_run(f"ghost-{uuid.uuid4().hex}")


# ---------------------------------------------------------------------------
# ensure_portfolio idempotency + buy integration
# ---------------------------------------------------------------------------


def test_ensure_portfolio_then_buy_debits_cash_and_creates_rows(
    migrated_engine: Engine,
) -> None:
    """ensure_portfolio (x2) is idempotent; a subsequent buy debits cash and
    creates exactly one holding row, one lot row, and one trade row.

    This is the regression test for the HITL fill bug:
      KeyError 'Portfolio <id> not found' fired because no portfolio row existed.
    Steps:
      1. Call ensure_portfolio twice with the same id — must produce one row.
      2. Call get_portfolio — must return a Portfolio with the starting cash.
      3. Call buy — must succeed (no KeyError).
      4. Call get_portfolio again — cash must be debited.
      5. Assert one trade row and one lot row exist in the DB.
    """
    from sqlalchemy import text

    starting_cash = Decimal("100000.00")
    portfolio_id = uuid.uuid4()
    repo = LedgerRepository(migrated_engine)

    # Step 1: idempotent seeding
    repo.ensure_portfolio(portfolio_id, starting_cash)
    repo.ensure_portfolio(portfolio_id, starting_cash)

    with migrated_engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM portfolios WHERE id = :id"),
            {"id": str(portfolio_id)},
        ).scalar_one()
    assert count == 1, "Two ensure_portfolio calls must produce exactly one row"

    # Step 2: get_portfolio reflects persisted cash
    portfolio_before = repo.get_portfolio(portfolio_id)
    assert portfolio_before.cash == starting_cash

    # Step 3: buy must not raise KeyError
    trade = _build_trade(portfolio_id, qty=Decimal("10"), price=Decimal("150.00"))
    filled = repo.buy(trade, portfolio_id)
    assert filled.status == TradeStatus.FILLED

    # Step 4: cash is debited
    portfolio_after = repo.get_portfolio(portfolio_id)
    assert portfolio_after.cash < starting_cash, "Cash must be debited after buy"

    # Step 5: one trade row and one lot row
    with migrated_engine.connect() as conn:
        trade_count = conn.execute(
            text("SELECT COUNT(*) FROM trades WHERE id = :id"),
            {"id": str(filled.id)},
        ).scalar_one()
        lot_count = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM lots l
                JOIN holdings h ON h.id = l.holding_id
                WHERE h.portfolio_id = :pid AND h.symbol = 'AAPL'
                """
            ),
            {"pid": str(portfolio_id)},
        ).scalar_one()

    assert trade_count == 1, "Exactly one trade row must exist after buy"
    assert lot_count == 1, "Exactly one lot row must exist after buy"
