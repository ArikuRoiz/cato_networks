"""Initial schema: portfolios, holdings, lots, trades, approvals, decision_cycles,
evidence, audit_log.

Revision ID: 0001
Revises:
Create Date: 2024-10-21 00:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # portfolios
    # ------------------------------------------------------------------
    op.create_table(
        "portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cash_balance", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_portfolios"),
    )

    # ------------------------------------------------------------------
    # holdings
    # ------------------------------------------------------------------
    op.create_table(
        "holdings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["portfolios.id"],
            name="fk_holdings_portfolio_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_holdings"),
        sa.UniqueConstraint("portfolio_id", "symbol", name="uq_holdings_portfolio_symbol"),
    )
    op.create_index("ix_holdings_portfolio_id", "holdings", ["portfolio_id"])

    # ------------------------------------------------------------------
    # lots
    # ------------------------------------------------------------------
    op.create_table(
        "lots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("holding_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("price", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["holding_id"],
            ["holdings.id"],
            name="fk_lots_holding_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lots"),
    )
    op.create_index("ix_lots_holding_id", "lots", ["holding_id"])

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------
    op.create_table(
        "trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("qty", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_price", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("fill_price", sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column("slippage", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("commission", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["portfolios.id"],
            name="fk_trades_portfolio_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trades"),
        sa.UniqueConstraint("idempotency_key", name="uq_trades_idempotency_key"),
    )
    op.create_index("ix_trades_cycle_id", "trades", ["cycle_id"])
    op.create_index("ix_trades_portfolio_id", "trades", ["portfolio_id"])

    # ------------------------------------------------------------------
    # approvals
    # ------------------------------------------------------------------
    op.create_table(
        "approvals",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trade_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("threshold_breached", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("decided_by", sa.String(256), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["trade_id"],
            ["trades.id"],
            name="fk_approvals_trade_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_approvals"),
    )
    op.create_index("ix_approvals_trade_id", "approvals", ["trade_id"])

    # ------------------------------------------------------------------
    # decision_cycles
    # ------------------------------------------------------------------
    op.create_table(
        "decision_cycles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trigger_type", sa.String(32), nullable=False),
        sa.Column("trigger_ref", sa.String(512), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_decision_cycles"),
    )
    op.create_index("ix_decision_cycles_started_at", "decision_cycles", ["started_at"])

    # ------------------------------------------------------------------
    # evidence
    # ------------------------------------------------------------------
    op.create_table(
        "evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_url", sa.Text, nullable=False),
        sa.Column("chunk_id", sa.String(256), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["decision_cycles.id"],
            name="fk_evidence_cycle_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evidence"),
    )
    op.create_index("ix_evidence_cycle_id", "evidence", ["cycle_id"])

    # ------------------------------------------------------------------
    # audit_log — append-only; no update/delete triggers by convention
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor", sa.String(64), nullable=False),
        sa.Column("action", sa.String(256), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_correlation_id", "audit_log", ["correlation_id"])
    op.create_index("ix_audit_log_ts", "audit_log", ["ts"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("evidence")
    op.drop_table("decision_cycles")
    op.drop_table("approvals")
    op.drop_table("trades")
    op.drop_table("lots")
    op.drop_table("holdings")
    op.drop_table("portfolios")
