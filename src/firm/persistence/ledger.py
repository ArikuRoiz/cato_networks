"""LedgerRepository — concrete Postgres repository for the firm's ledger."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload

from firm.domain import (
    Holding,
    InsufficientCash,
    InsufficientHolding,
    Lot,
    Portfolio,
    Trade,
    TradeStatus,
)
from firm.domain.portfolio import _COMMISSION_PER_SHARE, _SLIPPAGE_BPS
from firm.persistence.models import (
    AuditLogRow,
    HoldingRow,
    LotRow,
    PortfolioRow,
    TradeRow,
)


class LedgerRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_portfolio(self, portfolio_id: uuid.UUID) -> Portfolio:
        with Session(self._engine) as session:
            row = _load_portfolio_row(session, portfolio_id)
            return _portfolio_from_row(row)

    def get_trade(self, trade_id: uuid.UUID) -> Trade | None:
        with Session(self._engine) as session:
            row = session.get(TradeRow, trade_id)
            return _trade_from_row(row) if row is not None else None

    def get_trades_for_cycle(self, cycle_id: uuid.UUID) -> list[Trade]:
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
        return self._run_in_transaction(
            trade,
            portfolio_id,
            lambda s, t, p: _execute_buy(s, t, p, opened_at=opened_at),
        )

    def sell(self, trade: Trade, portfolio_id: uuid.UUID) -> Trade:
        return self._run_in_transaction(trade, portfolio_id, _execute_sell)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_in_transaction(
        self,
        trade: Trade,
        portfolio_id: uuid.UUID,
        executor: Callable[[Session, Trade, uuid.UUID], Trade],
    ) -> Trade:
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            existing = _find_by_idempotency_key(session, trade.idempotency_key)
            if existing is not None:
                return existing
            filled = executor(session, trade, portfolio_id)
            _append_audit(session, filled.cycle_id, "system", "trade.filled", filled.model_dump(mode="json"))
            session.commit()
            return filled


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _execute_buy(
    session: Session,
    trade: Trade,
    portfolio_id: uuid.UUID,
    opened_at: datetime | None = None,
) -> Trade:
    portfolio_row = _load_portfolio_row(session, portfolio_id)
    fill_price = trade.requested_price * (Decimal("1") + _SLIPPAGE_BPS)
    commission = trade.qty * _COMMISSION_PER_SHARE
    _debit_cash(portfolio_row, trade.qty * fill_price + commission)
    _insert_lot(session, _upsert_holding(session, portfolio_row, trade.symbol), trade, fill_price, opened_at=opened_at)
    filled = _fill_trade(trade, fill_price, commission)
    return _trade_from_row(_insert_trade_row(session, filled, portfolio_id))


def _execute_sell(
    session: Session,
    trade: Trade,
    portfolio_id: uuid.UUID,
) -> Trade:
    portfolio_row = _load_portfolio_row(session, portfolio_id)
    holding = _portfolio_from_row(portfolio_row).holdings.get(trade.symbol)
    if holding is None:
        raise InsufficientHolding(f"Cannot sell {trade.symbol}; no position held.")
    fill_price = trade.requested_price * (Decimal("1") - _SLIPPAGE_BPS)
    commission = trade.qty * _COMMISSION_PER_SHARE
    _apply_lot_closures(session, holding.close_lots(trade.qty))
    _delete_holding_if_empty(session, portfolio_row, trade.symbol)
    portfolio_row.cash_balance += trade.qty * fill_price - commission
    filled = _fill_trade(trade, fill_price, commission)
    return _trade_from_row(_insert_trade_row(session, filled, portfolio_id))


def _load_portfolio_row(session: Session, portfolio_id: uuid.UUID) -> PortfolioRow:
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
    stmt = select(TradeRow).where(TradeRow.idempotency_key == key)
    row = session.execute(stmt).scalar_one_or_none()
    return _trade_from_row(row) if row is not None else None


def _debit_cash(portfolio_row: PortfolioRow, amount: Decimal) -> None:
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
    for h in portfolio_row.holdings:
        if h.symbol == symbol:
            return h
    holding = HoldingRow(id=uuid.uuid4(), portfolio_id=portfolio_row.id, symbol=symbol)
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
    for h in list(portfolio_row.holdings):
        if h.symbol == symbol and not h.lots:
            session.delete(h)
            portfolio_row.holdings.remove(h)
            break


def _fill_trade(trade: Trade, fill_price: Decimal, commission: Decimal) -> Trade:
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
    row = AuditLogRow(
        id=uuid.uuid4(),
        correlation_id=correlation_id,
        actor=actor,
        action=action,
        payload=payload,
        ts=datetime.now(tz=UTC),
    )
    session.add(row)
