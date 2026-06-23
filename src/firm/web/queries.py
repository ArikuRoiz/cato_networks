"""Read-only database queries for the web dashboard.

All functions consume a SQLAlchemy engine and return typed DTOs.
No SQLAlchemy sessions or ORM objects cross module boundaries.

Module order: public query functions → private row-to-DTO converters → SQL helpers.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload

from firm.persistence.models import (
    AuditLogRow,
    DecisionCycleRow,
    HoldingRow,
    PortfolioRow,
    TradeRow,
)
from firm.web.schemas import (
    AuditEntryDTO,
    CycleDTO,
    HoldingDTO,
    PortfolioDTO,
    TradeDTO,
    compute_nav_and_pnl,
)

# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def fetch_portfolio(engine: Engine, portfolio_id: uuid.UUID) -> PortfolioDTO:
    """Return portfolio summary for *portfolio_id*."""
    with Session(engine) as session:
        row = _load_portfolio_row(session, portfolio_id)
        return _portfolio_dto_from_row(row)


def fetch_recent_trades(engine: Engine, limit: int = 50) -> list[TradeDTO]:
    """Return the *limit* most-recently filled trades."""
    with Session(engine) as session:
        stmt = select(TradeRow).order_by(TradeRow.filled_at.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [_trade_dto_from_row(r) for r in rows]


def fetch_recent_cycles(engine: Engine, limit: int = 50) -> list[CycleDTO]:
    """Return the *limit* most-recent decision cycles with judge metadata."""
    with Session(engine) as session:
        rows = _query_cycles_with_judge(session, limit)
        return [_cycle_dto_from_row(r) for r in rows]


def fetch_cycle_trace(engine: Engine, correlation_id: str) -> list[AuditEntryDTO]:
    """Return ordered audit_log entries for *correlation_id*."""
    try:
        corr_uuid = uuid.UUID(correlation_id)
    except ValueError:
        return []
    with Session(engine) as session:
        stmt = (
            select(AuditLogRow)
            .where(AuditLogRow.correlation_id == corr_uuid)
            .order_by(AuditLogRow.ts)
        )
        rows = session.execute(stmt).scalars().all()
        return [_audit_dto_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Private row-to-DTO converters
# ---------------------------------------------------------------------------


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


def _portfolio_dto_from_row(row: PortfolioRow) -> PortfolioDTO:
    holding_tuples = _extract_holding_tuples(row)
    holding_dtos = [
        HoldingDTO(symbol=sym, quantity=str(qty), avg_cost=str(cost))
        for sym, qty, cost in holding_tuples
    ]
    nav, pnl = compute_nav_and_pnl(row.cash_balance, holding_tuples)
    return PortfolioDTO(
        cash=str(row.cash_balance),
        nav=str(nav),
        pnl=str(pnl),
        holdings=holding_dtos,
    )


def _extract_holding_tuples(
    row: PortfolioRow,
) -> list[tuple[str, Decimal, Decimal]]:
    """Return (symbol, total_qty, avg_cost) for each holding."""
    result = []
    for h in row.holdings:
        total_qty = sum((lot.quantity for lot in h.lots), Decimal("0"))
        if total_qty <= Decimal("0"):
            continue
        total_cost = sum((lot.quantity * lot.price for lot in h.lots), Decimal("0"))
        avg_cost = total_cost / total_qty
        result.append((h.symbol, total_qty, avg_cost))
    return result


def _trade_dto_from_row(row: TradeRow) -> TradeDTO:
    return TradeDTO(
        id=str(row.id),
        symbol=row.symbol,
        side=row.side,
        qty=str(row.qty),
        status=row.status,
        fill_price=str(row.fill_price) if row.fill_price is not None else None,
        filled_at=row.filled_at.isoformat() if row.filled_at is not None else None,
    )


def _query_cycles_with_judge(session: Session, limit: int) -> list[Any]:
    """Return recent decision_cycles rows joined with the cycle outcome payload."""
    stmt = select(DecisionCycleRow).order_by(DecisionCycleRow.started_at.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def _cycle_dto_from_row(row: DecisionCycleRow) -> CycleDTO:
    return CycleDTO(
        id=str(row.id),
        correlation_id=row.trigger_ref,
        trigger_type=row.trigger_type,
        outcome=row.outcome,
        started_at=row.started_at.isoformat(),
        # Judge metadata is in the audit_log payload; surface None here —
        # the frontend can request the trace to get full detail.
        judge_score=None,
        alignment=None,
    )


def _audit_dto_from_row(row: AuditLogRow) -> AuditEntryDTO:
    return AuditEntryDTO(
        id=str(row.id),
        correlation_id=str(row.correlation_id),
        actor=row.actor,
        action=row.action,
        payload=dict(row.payload),
        ts=row.ts.isoformat(),
    )
