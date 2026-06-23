"""LedgerRepository — concrete Postgres repository for the firm's ledger."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
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
    ApprovalRow,
    AuditLogRow,
    DecisionCycleRow,
    HoldingRow,
    LotRow,
    PendingRunRow,
    PortfolioRow,
    TradeRow,
)

# ---------------------------------------------------------------------------
# Stable firm portfolio identity
# ---------------------------------------------------------------------------

#: One fixed UUID shared by the CLI, web runtime, and every live graph run.
#: A new UUID each run would make the ledger write to a fresh, empty portfolio
#: on every restart — cash never survives between runs.  This constant is the
#: single source of truth; import it wherever a live portfolio_id is needed.
FIRM_PORTFOLIO_ID: uuid.UUID = uuid.UUID("00000000-0000-4000-8000-000000000001")

# ---------------------------------------------------------------------------
# CycleAuditRecord — typed input to record_cycle; wraps at the boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleAuditRecord:
    """All information captured at the end of a decision cycle.

    Callers construct this from GraphState fields; the repository converts it
    into one ``decision_cycles`` row + several ``audit_log`` rows atomically.

    Fields:
        correlation_id: UUID string copied from GraphState["correlation_id"].
        symbol: Ticker symbol being evaluated.
        trigger_type: "scheduled" | "event".
        decision_ts: Timestamp when the cycle started (UTC).
        recommendation: Recommendation string from the research plan, or None.
        conviction: Research manager conviction score, or None.
        outcome: CycleOutcome string (filled/hold/rejected/…).
        judge_score: Coherence score from JudgeAgent (1-5), or None.
        alignment: VerdictAlignment string from JudgeAgent, or None.
        trade_id: UUID of the filled Trade if outcome is "filled", else None.
    """

    correlation_id: str
    symbol: str
    trigger_type: str
    decision_ts: datetime
    recommendation: str | None
    conviction: float | None
    outcome: str
    judge_score: int | None
    alignment: str | None
    trade_id: uuid.UUID | None


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

    def ensure_portfolio(
        self,
        portfolio_id: uuid.UUID,
        starting_cash: Decimal,
    ) -> None:
        """Idempotently create a portfolio row if one does not already exist.

        Uses INSERT … ON CONFLICT DO NOTHING so repeated calls are safe.  Call
        this once at server/pipeline startup so ``GET /api/portfolio`` always
        finds a row rather than returning zeros.

        Args:
            portfolio_id: The stable UUID for this portfolio.
            starting_cash: Initial cash balance (e.g. 100 000).
        """
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            _insert_portfolio_if_absent(session, portfolio_id, starting_cash)
            session.commit()

    # ------------------------------------------------------------------
    # Pending-run registry (web HITL thread tracking)
    # ------------------------------------------------------------------

    def register_pending_run(
        self,
        *,
        thread_id: str,
        correlation_id: str,
        symbol: str,
    ) -> None:
        """Record a background graph run so pending_approvals() can find it.

        Idempotent: repeated calls with the same thread_id are silently ignored
        (ON CONFLICT DO NOTHING at the DB level via a try/except on flush).

        Args:
            thread_id: LangGraph thread ID used as the checkpoint namespace.
            correlation_id: Cycle correlation UUID string.
            symbol: Ticker being analysed.
        """
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            _insert_pending_run(session, thread_id, correlation_id, symbol)
            session.commit()

    def delete_pending_run(self, thread_id: str) -> None:
        """Remove a pending-run entry once the thread has been resumed or completed.

        No-op when the thread_id is not found (safe to call on every resume).

        Args:
            thread_id: LangGraph thread ID to remove from the registry.
        """
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            _delete_pending_run(session, thread_id)
            session.commit()

    def list_pending_runs(self) -> list[tuple[str, str, str]]:
        """Return all registered pending runs as (thread_id, correlation_id, symbol) tuples.

        Used by pending_approvals() to enumerate threads that may be paused on
        a HITL interrupt.  Returns an empty list when no runs are registered.
        """
        with Session(self._engine) as session:
            rows = session.execute(select(PendingRunRow)).scalars().all()
            return [(r.thread_id, r.correlation_id, r.symbol) for r in rows]

    def record_approval(
        self,
        *,
        correlation_id: uuid.UUID,
        trade_id: uuid.UUID,
        status: str,
        original_notional: Decimal,
        original_qty: Decimal,
        edited_qty: Decimal | None = None,
        decided_at: datetime | None = None,
        decided_by: str = "risk_committee",
    ) -> None:
        """Write an ApprovalRow and matching audit entry in a single transaction.

        Captures the full HITL decision so the firm can audit and replay every
        human override.  The ApprovalRow is the navigable FK record; the
        audit_log entry carries the rich financial payload for full replayability.

        Args:
            correlation_id: Cycle correlation UUID (used as FK in audit_log).
            trade_id: FK to the TradeRow being approved/rejected.
            status: One of 'approved', 'rejected', 'expired' (HITLStatus values).
            original_notional: Notional value of the original proposal.
            original_qty: Quantity of the original proposal.
            edited_qty: Human-adjusted quantity; None when not edited.
            decided_at: Timestamp of the decision; defaults to now(UTC).
            decided_by: Identity of the decision-maker; defaults to 'risk_committee'.
        """
        ts = decided_at if decided_at is not None else datetime.now(UTC)
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            _insert_approval_row(
                session,
                trade_id=trade_id,
                status=status,
                decided_by=decided_by,
                decided_at=ts,
            )
            _append_audit(
                session,
                correlation_id=correlation_id,
                actor=decided_by,
                action="hitl.decision",
                payload=_approval_audit_payload(
                    correlation_id=correlation_id,
                    trade_id=trade_id,
                    status=status,
                    original_notional=original_notional,
                    original_qty=original_qty,
                    edited_qty=edited_qty,
                    decided_at=ts,
                    decided_by=decided_by,
                ),
            )
            session.commit()

    def record_cycle(self, record: CycleAuditRecord) -> None:
        """Persist a decision cycle and its key audit steps atomically.

        Writes one ``decision_cycles`` row and four ``audit_log`` entries in a
        single transaction, regardless of cycle outcome (hold, rejected, filled,
        error).  Designed to be called at the end of every cycle path so no
        decision is invisible in the DB.

        Args:
            record: Fully-populated CycleAuditRecord from the reporting node.
        """
        correlation_uuid = uuid.UUID(record.correlation_id)
        cycle_row_id = uuid.uuid4()
        with Session(self._engine, autobegin=False) as session:
            session.begin()
            _insert_decision_cycle_row(session, cycle_row_id, record)
            _append_audit(
                session,
                correlation_id=correlation_uuid,
                actor="research_manager",
                action="research.done",
                payload=_research_done_payload(record),
            )
            _append_audit(
                session,
                correlation_id=correlation_uuid,
                actor="pm",
                action="decision.made",
                payload=_decision_made_payload(record),
            )
            _append_audit(
                session,
                correlation_id=correlation_uuid,
                actor="risk",
                action="risk.outcome",
                payload=_risk_outcome_payload(record),
            )
            _append_audit(
                session,
                correlation_id=correlation_uuid,
                actor="system",
                action="cycle.outcome",
                payload=_cycle_outcome_payload(record, cycle_row_id),
            )
            session.commit()

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
            _append_audit(
                session, filled.cycle_id, "system", "trade.filled", filled.model_dump(mode="json")
            )
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
    _insert_lot(
        session,
        _upsert_holding(session, portfolio_row, trade.symbol),
        trade,
        fill_price,
        opened_at=opened_at,
    )
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


def _insert_approval_row(
    session: Session,
    *,
    trade_id: uuid.UUID,
    status: str,
    decided_by: str,
    decided_at: datetime,
) -> None:
    row = ApprovalRow(
        id=uuid.uuid4(),
        trade_id=trade_id,
        threshold_breached="hitl_threshold_pct",
        status=status,
        decided_by=decided_by,
        expires_at=decided_at,  # approval window already elapsed; store decided_at
        decided_at=decided_at,
    )
    session.add(row)


def _approval_audit_payload(
    *,
    correlation_id: uuid.UUID,
    trade_id: uuid.UUID,
    status: str,
    original_notional: Decimal,
    original_qty: Decimal,
    edited_qty: Decimal | None,
    decided_at: datetime,
    decided_by: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": str(correlation_id),
        "trade_id": str(trade_id),
        "status": status,
        "original_notional": str(original_notional),
        "original_qty": str(original_qty),
        "decided_at": decided_at.isoformat(),
        "decided_by": decided_by,
    }
    if edited_qty is not None:
        payload["edited_qty"] = str(edited_qty)
    return payload


# ---------------------------------------------------------------------------
# record_cycle helpers — one function per audit payload, no inline dicts
# ---------------------------------------------------------------------------


def _insert_decision_cycle_row(
    session: Session,
    cycle_row_id: uuid.UUID,
    record: CycleAuditRecord,
) -> None:
    row = DecisionCycleRow(
        id=cycle_row_id,
        trigger_type=record.trigger_type,
        trigger_ref=record.correlation_id,
        started_at=record.decision_ts,
        outcome=record.outcome,
    )
    session.add(row)


def _research_done_payload(record: CycleAuditRecord) -> dict[str, Any]:
    return {
        "correlation_id": record.correlation_id,
        "symbol": record.symbol,
        "recommendation": record.recommendation,
        "conviction": record.conviction,
    }


def _decision_made_payload(record: CycleAuditRecord) -> dict[str, Any]:
    return {
        "correlation_id": record.correlation_id,
        "symbol": record.symbol,
        "recommendation": record.recommendation,
        "conviction": record.conviction,
        "decision_ts": record.decision_ts.isoformat(),
    }


def _risk_outcome_payload(record: CycleAuditRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": record.correlation_id,
        "symbol": record.symbol,
        "outcome": record.outcome,
    }
    if record.trade_id is not None:
        payload["trade_id"] = str(record.trade_id)
    return payload


def _cycle_outcome_payload(
    record: CycleAuditRecord,
    cycle_row_id: uuid.UUID,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": record.correlation_id,
        "symbol": record.symbol,
        "outcome": record.outcome,
        "cycle_row_id": str(cycle_row_id),
        "decision_ts": record.decision_ts.isoformat(),
    }
    if record.judge_score is not None:
        payload["judge_score"] = record.judge_score
    if record.alignment is not None:
        payload["alignment"] = record.alignment
    if record.trade_id is not None:
        payload["trade_id"] = str(record.trade_id)
    return payload


# ---------------------------------------------------------------------------
# ensure_portfolio + pending_run helpers
# ---------------------------------------------------------------------------


def _insert_portfolio_if_absent(
    session: Session,
    portfolio_id: uuid.UUID,
    starting_cash: Decimal,
) -> None:
    """INSERT a PortfolioRow only when the given portfolio_id does not exist.

    SQLAlchemy does not expose ON CONFLICT DO NOTHING natively; we emulate it
    with a SELECT-then-INSERT pattern inside a single transaction so concurrent
    callers are safe (the PK constraint acts as the guard at the DB level).
    """
    from sqlalchemy import text

    session.execute(
        text(
            """
            INSERT INTO portfolios (id, cash_balance, created_at)
            VALUES (:id, :cash, NOW())
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {"id": str(portfolio_id), "cash": str(starting_cash)},
    )


def _insert_pending_run(
    session: Session,
    thread_id: str,
    correlation_id: str,
    symbol: str,
) -> None:
    from sqlalchemy import text

    session.execute(
        text(
            """
            INSERT INTO pending_runs (thread_id, correlation_id, symbol, started_at)
            VALUES (:tid, :cid, :sym, NOW())
            ON CONFLICT (thread_id) DO NOTHING
            """
        ),
        {"tid": thread_id, "cid": correlation_id, "sym": symbol},
    )


def _delete_pending_run(session: Session, thread_id: str) -> None:
    from sqlalchemy import text

    session.execute(
        text("DELETE FROM pending_runs WHERE thread_id = :tid"),
        {"tid": thread_id},
    )
