"""LedgerRepository — concrete Postgres repository for the firm's ledger.

This is intentionally NOT a port.  It is a concrete implementation tested against
a real (ephemeral) Postgres instance via testcontainers.  It owns the single ACID
boundary for all money writes.

Design constraints:
- buy() and sell() each execute in ONE transaction.
- Idempotency key checked first; duplicate returns the existing trade unchanged.
- AuditLogRow appended inside the same transaction as the money write.
- No defensive isinstance checks; callers pass well-typed domain objects.
- Functions ≤ 30 lines; helpers extract the sub-steps.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload

from firm.domain.entities import (
    _COMMISSION_PER_SHARE,
    _SLIPPAGE_BPS,
    DecisionCycle,
    Holding,
    InsufficientCash,
    InsufficientHolding,
    Lot,
    Portfolio,
    Trade,
    TradeStatus,
)
from firm.persistence.models import (
    AuditLogRow,
    DecisionCycleRow,
    HoldingRow,
    LotRow,
    PortfolioRow,
    TradeRow,
)

# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class LedgerRepository:
    """Transactional ledger backed by Postgres.

    ``engine`` must already point at the migrated database.  Call
    ``LedgerRepository(engine)`` and inject where needed; never construct
    an engine inside this class.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_portfolio(self, portfolio_id: uuid.UUID) -> Portfolio:
        """Reconstruct a Portfolio domain object from the database.

        Loads PortfolioRow → HoldingRow → LotRow in a single session so
        all lazy relationships are resolved before the session closes.
        Raises ``KeyError`` when *portfolio_id* is not found.
        """
        with Session(self._engine) as session:
            row = _load_portfolio_row(session, portfolio_id)
            return _portfolio_from_row(row)

    def get_trade(self, trade_id: uuid.UUID) -> Trade | None:
        """Return the Trade for *trade_id*, or None if not found."""
        with Session(self._engine) as session:
            row = session.get(TradeRow, trade_id)
            if row is None:
                return None
            return _trade_from_row(row)

    def get_trades_for_cycle(self, cycle_id: uuid.UUID) -> list[Trade]:
        """Return all filled trades belonging to *cycle_id*."""
        with Session(self._engine) as session:
            stmt = select(TradeRow).where(TradeRow.cycle_id == cycle_id)
            rows = session.execute(stmt).scalars().all()
            return [_trade_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    def buy(
        self,
        trade: Trade,
        portfolio_id: uuid.UUID,
        opened_at: datetime | None = None,
    ) -> Trade:
        """Execute a buy in one ACID transaction.

        Checks idempotency_key first — returns the existing fill if duplicate.
        Otherwise: debit cash, insert HoldingRow/LotRow, insert TradeRow(FILLED),
        append AuditLogRow; all in a single BEGIN/COMMIT.

        ``opened_at`` sets the lot's cost-basis timestamp; supply an explicit
        value in tests to make FIFO ordering deterministic without mocking the
        datetime class.
        """
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            existing = _find_by_idempotency_key(session, trade.idempotency_key)
            if existing is not None:
                return existing
            result = _execute_buy(session, trade, portfolio_id, opened_at=opened_at)
            session.commit()
            return result

    def sell(self, trade: Trade, portfolio_id: uuid.UUID) -> Trade:
        """Execute a sell in one ACID transaction.

        Checks idempotency_key first — returns the existing fill if duplicate.
        Otherwise: FIFO lot closures, credit cash, insert TradeRow(FILLED),
        append AuditLogRow; all in a single BEGIN/COMMIT.
        """
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            existing = _find_by_idempotency_key(session, trade.idempotency_key)
            if existing is not None:
                return existing
            result = _execute_sell(session, trade, portfolio_id)
            session.commit()
            return result

    # ------------------------------------------------------------------
    # Decision cycle + audit
    # ------------------------------------------------------------------

    def record_decision_cycle(self, cycle: DecisionCycle) -> None:
        """Persist a DecisionCycle record (upsert by id)."""
        with Session(self._engine) as session:
            row = session.get(DecisionCycleRow, cycle.id)
            if row is None:
                row = DecisionCycleRow(
                    id=cycle.id,
                    trigger_type=cycle.trigger_type,
                    trigger_ref=cycle.trigger_ref,
                    started_at=cycle.started_at,
                    outcome=cycle.outcome,
                )
                session.add(row)
            else:
                row.outcome = cycle.outcome
            session.commit()

    def append_audit(
        self,
        correlation_id: uuid.UUID,
        actor: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        """Append one row to the append-only audit log."""
        with Session(self._engine) as session:
            _append_audit(session, correlation_id, actor, action, payload)
            session.commit()


# ---------------------------------------------------------------------------
# Private helpers — each ≤ 30 lines, named for intent
# ---------------------------------------------------------------------------


def _execute_buy(
    session: Session,
    trade: Trade,
    portfolio_id: uuid.UUID,
    opened_at: datetime | None = None,
) -> Trade:
    """Inner buy body — must run inside an open transaction."""
    portfolio_row = _load_portfolio_row(session, portfolio_id)
    fill_price = _apply_slippage_buy(trade.requested_price)
    notional = trade.qty * fill_price
    commission = trade.qty * _COMMISSION_PER_SHARE
    _debit_cash(portfolio_row, notional + commission)
    holding_row = _upsert_holding(session, portfolio_row, trade.symbol)
    _insert_lot(session, holding_row, trade, fill_price, opened_at=opened_at)
    filled = _fill_trade(trade, fill_price, commission)
    trade_row = _insert_trade_row(session, filled, portfolio_id)
    _append_audit(
        session,
        correlation_id=trade.cycle_id,
        actor="system",
        action="trade.filled",
        payload=_trade_audit_payload(filled),
    )
    return _trade_from_row(trade_row)


def _execute_sell(
    session: Session,
    trade: Trade,
    portfolio_id: uuid.UUID,
) -> Trade:
    """Inner sell body — must run inside an open transaction."""
    portfolio_row = _load_portfolio_row(session, portfolio_id)
    portfolio = _portfolio_from_row(portfolio_row)
    holding = portfolio.holdings.get(trade.symbol)
    if holding is None:
        raise InsufficientHolding(f"Cannot sell {trade.symbol}; no position held.")
    lot_pairs = holding.close_lots(trade.qty)
    fill_price = _apply_slippage_sell(trade.requested_price)
    commission = trade.qty * _COMMISSION_PER_SHARE
    proceeds = trade.qty * fill_price - commission
    _apply_lot_closures(session, lot_pairs)
    _delete_holding_if_empty(session, portfolio_row, trade.symbol)
    portfolio_row.cash_balance = portfolio_row.cash_balance + proceeds
    filled = _fill_trade(trade, fill_price, commission)
    trade_row = _insert_trade_row(session, filled, portfolio_id)
    _append_audit(
        session,
        correlation_id=trade.cycle_id,
        actor="system",
        action="trade.filled",
        payload=_trade_audit_payload(filled),
    )
    return _trade_from_row(trade_row)


def _load_portfolio_row(session: Session, portfolio_id: uuid.UUID) -> PortfolioRow:
    """Load PortfolioRow with holdings and lots eagerly; raise KeyError if absent."""
    stmt = (
        select(PortfolioRow)
        .where(PortfolioRow.id == portfolio_id)
        .options(selectinload(PortfolioRow.holdings).selectinload(HoldingRow.lots))
    )
    row = session.execute(stmt).scalar_one_or_none()
    if row is None:
        raise KeyError(f"Portfolio {portfolio_id} not found.")
    return row


def _portfolio_from_row(row: PortfolioRow) -> Portfolio:
    """Reconstruct a Portfolio domain object from an ORM row."""
    holdings: dict[str, Holding] = {}
    for h_row in row.holdings:
        lots = [
            Lot(
                id=lot.id,
                symbol=lot.symbol,
                qty=lot.quantity,
                cost=lot.price,
                opened_at=lot.opened_at,
            )
            for lot in h_row.lots
        ]
        holdings[h_row.symbol] = Holding(symbol=h_row.symbol, lots=lots)
    return Portfolio(id=row.id, cash=row.cash_balance, holdings=holdings)


def _trade_from_row(row: TradeRow) -> Trade:
    """Reconstruct a Trade domain object from an ORM row."""
    return Trade(
        id=row.id,
        cycle_id=row.cycle_id,
        symbol=row.symbol,
        side=row.side,  # type: ignore[arg-type]
        qty=row.qty,
        status=TradeStatus(row.status),
        requested_price=row.requested_price,
        fill_price=row.fill_price,
        slippage=row.slippage,
        commission=row.commission,
        idempotency_key=row.idempotency_key,
    )


def _find_by_idempotency_key(session: Session, key: str) -> Trade | None:
    """Return an existing Trade if *key* was already filled; None otherwise."""
    stmt = select(TradeRow).where(TradeRow.idempotency_key == key)
    row = session.execute(stmt).scalar_one_or_none()
    return _trade_from_row(row) if row is not None else None


def _apply_slippage_buy(price: Decimal) -> Decimal:
    """Fill price for a buy: mid + slippage bps."""
    return price * (Decimal("1") + _SLIPPAGE_BPS)


def _apply_slippage_sell(price: Decimal) -> Decimal:
    """Fill price for a sell: mid minus slippage bps."""
    return price * (Decimal("1") - _SLIPPAGE_BPS)


def _debit_cash(portfolio_row: PortfolioRow, amount: Decimal) -> None:
    """Debit *amount* from portfolio cash; raise InsufficientCash if short."""
    if portfolio_row.cash_balance < amount:
        raise InsufficientCash(
            f"Cash {portfolio_row.cash_balance} insufficient for debit {amount}."
        )
    portfolio_row.cash_balance = portfolio_row.cash_balance - amount


def _upsert_holding(
    session: Session,
    portfolio_row: PortfolioRow,
    symbol: str,
) -> HoldingRow:
    """Return existing HoldingRow for *symbol* or insert a new one."""
    for h in portfolio_row.holdings:
        if h.symbol == symbol:
            return h
    holding = HoldingRow(
        id=uuid.uuid4(),
        portfolio_id=portfolio_row.id,
        symbol=symbol,
    )
    session.add(holding)
    portfolio_row.holdings.append(holding)
    return holding


def _insert_lot(
    session: Session,
    holding_row: HoldingRow,
    trade: Trade,
    fill_price: Decimal,
    opened_at: datetime | None = None,
) -> None:
    """Insert a new LotRow for the buy; cost basis is the actual fill price.

    ``opened_at`` defaults to ``datetime.now(UTC)``; supply an explicit value
    in tests to guarantee deterministic FIFO ordering across lots.
    """
    lot = LotRow(
        id=uuid.uuid4(),
        holding_id=holding_row.id,
        symbol=trade.symbol,
        quantity=trade.qty,
        price=fill_price,
        opened_at=opened_at if opened_at is not None else datetime.now(tz=UTC),
    )
    session.add(lot)


def _apply_lot_closures(
    session: Session,
    lot_pairs: list[tuple[Lot, Decimal]],
) -> None:
    """Reduce or delete LotRows according to FIFO lot-pair closures."""
    for lot, consumed_qty in lot_pairs:
        row = session.get(LotRow, lot.id)
        if row is None:
            continue
        remaining = lot.qty - consumed_qty
        if remaining <= Decimal("0"):
            session.delete(row)
        else:
            row.quantity = remaining


def _delete_holding_if_empty(
    session: Session,
    portfolio_row: PortfolioRow,
    symbol: str,
) -> None:
    """Delete the HoldingRow for *symbol* when all its lots have been consumed.

    Keeps the schema consistent: a HoldingRow with no child LotRows would
    reconstruct as a zero-quantity Holding in the domain model, polluting the
    portfolio's holdings dict with ghost positions.
    """
    for h in list(portfolio_row.holdings):
        if h.symbol == symbol and not h.lots:
            session.delete(h)
            portfolio_row.holdings.remove(h)
            break


def _fill_trade(trade: Trade, fill_price: Decimal, commission: Decimal) -> Trade:
    """Return a copy of *trade* with FILLED status and fill metrics set."""
    slippage = abs(fill_price - trade.requested_price) * trade.qty
    return trade.model_copy(
        update={
            "status": TradeStatus.FILLED,
            "fill_price": fill_price,
            "slippage": slippage,
            "commission": commission,
        }
    )


def _insert_trade_row(
    session: Session,
    trade: Trade,
    portfolio_id: uuid.UUID,
) -> TradeRow:
    """Insert and return a TradeRow for *trade*."""
    row = TradeRow(
        id=trade.id,
        cycle_id=trade.cycle_id,
        portfolio_id=portfolio_id,
        symbol=trade.symbol,
        side=trade.side,
        qty=trade.qty,
        status=trade.status.value,
        requested_price=trade.requested_price,
        fill_price=trade.fill_price,
        slippage=trade.slippage,
        commission=trade.commission,
        idempotency_key=trade.idempotency_key,
        filled_at=datetime.now(tz=UTC),
    )
    session.add(row)
    return row


def _append_audit(
    session: Session,
    correlation_id: uuid.UUID,
    actor: str,
    action: str,
    payload: dict[str, Any],
) -> None:
    """Insert one AuditLogRow into the append-only log."""
    row = AuditLogRow(
        id=uuid.uuid4(),
        correlation_id=correlation_id,
        actor=actor,
        action=action,
        payload=payload,
        ts=datetime.now(tz=UTC),
    )
    session.add(row)


def _trade_audit_payload(trade: Trade) -> dict[str, Any]:
    """Serialise key trade fields into a JSON-safe dict for the audit log."""
    return {
        "trade_id": str(trade.id),
        "symbol": trade.symbol,
        "side": trade.side,
        "qty": str(trade.qty),
        "fill_price": str(trade.fill_price),
        "commission": str(trade.commission),
        "slippage": str(trade.slippage),
        "status": trade.status.value,
        "idempotency_key": trade.idempotency_key,
    }
