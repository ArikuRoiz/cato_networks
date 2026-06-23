"""SQLAlchemy ORM models for the firm's Postgres schema.

All monetary values use NUMERIC; all timestamps use TIMESTAMPTZ; all PKs are UUID.
AuditLogRow is append-only — no update or delete methods are exposed.
TradeRow.idempotency_key carries a unique constraint.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# PortfolioRow
# ---------------------------------------------------------------------------


class PortfolioRow(Base):
    __tablename__ = "portfolios"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=6), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    holdings: Mapped[list[HoldingRow]] = relationship(
        "HoldingRow", back_populates="portfolio", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# HoldingRow
# ---------------------------------------------------------------------------


class HoldingRow(Base):
    __tablename__ = "holdings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)

    portfolio: Mapped[PortfolioRow] = relationship("PortfolioRow", back_populates="holdings")
    lots: Mapped[list[LotRow]] = relationship(
        "LotRow", back_populates="holding", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", name="uq_holdings_portfolio_symbol"),
        Index("ix_holdings_portfolio_id", "portfolio_id"),
    )


# ---------------------------------------------------------------------------
# LotRow
# ---------------------------------------------------------------------------


class LotRow(Base):
    __tablename__ = "lots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    holding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("holdings.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=6), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    holding: Mapped[HoldingRow] = relationship("HoldingRow", back_populates="lots")

    __table_args__ = (Index("ix_lots_holding_id", "holding_id"),)


# ---------------------------------------------------------------------------
# TradeRow
# ---------------------------------------------------------------------------


class TradeRow(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cycle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="RESTRICT"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_price: Mapped[Decimal] = mapped_column(Numeric(precision=20, scale=6), nullable=False)
    fill_price: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=20, scale=6), nullable=True
    )
    slippage: Mapped[Decimal | None] = mapped_column(Numeric(precision=20, scale=8), nullable=True)
    commission: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=20, scale=8), nullable=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_trades_idempotency_key"),
        Index("ix_trades_cycle_id", "cycle_id"),
        Index("ix_trades_portfolio_id", "portfolio_id"),
    )


# ---------------------------------------------------------------------------
# ApprovalRow
# ---------------------------------------------------------------------------


class ApprovalRow(Base):
    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("trades.id", ondelete="CASCADE"), nullable=False
    )
    threshold_breached: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    decided_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_approvals_trade_id", "trade_id"),)


# ---------------------------------------------------------------------------
# DecisionCycleRow — one row per pipeline cycle, written by the reporting node
# ---------------------------------------------------------------------------


class DecisionCycleRow(Base):
    """Persistent record of a single decision cycle, regardless of outcome.

    Columns match the ``decision_cycles`` table created in migration 0001.
    ``trigger_ref`` carries the correlation_id string for cross-table tracing.
    ``outcome`` is a CycleOutcome string (filled/hold/rejected/…).
    """

    __tablename__ = "decision_cycles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (Index("ix_decision_cycles_started_at", "started_at"),)


# ---------------------------------------------------------------------------
# AuditLogRow — append-only (no update/delete methods exposed on this model)
# ---------------------------------------------------------------------------


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    correlation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(256), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_audit_log_correlation_id", "correlation_id"),
        Index("ix_audit_log_ts", "ts"),
    )
