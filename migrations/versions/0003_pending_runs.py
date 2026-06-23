"""Add pending_runs registry table for web HITL tracking.

Records thread_id + correlation_id + symbol for every background graph run
started via POST /api/run so pending_approvals() can enumerate paused threads
reliably without depending on PostgresSaver.list_namespaces().

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-23 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pending_runs",
        sa.Column("thread_id", sa.String(64), nullable=False),
        sa.Column("correlation_id", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("thread_id", name="pk_pending_runs"),
    )
    op.create_index("ix_pending_runs_started_at", "pending_runs", ["started_at"])


def downgrade() -> None:
    op.drop_index("ix_pending_runs_started_at", table_name="pending_runs")
    op.drop_table("pending_runs")
