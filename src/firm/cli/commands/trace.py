"""``firm trace`` — reconstruct the audit log for a single trade."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from firm.cli.output import _emit, _load_settings

logger = logging.getLogger(__name__)


def _cmd_trace(args: argparse.Namespace) -> None:
    """Reconstruct the audit log for a single trade by trade_id.

    Queries the audit_log table for all entries whose correlation_id matches
    the decision cycle that contains the given trade_id. Emits NDJSON to
    stdout (one event per line).
    """
    from firm.observability.tracing import (
        get_correlation_id,
        reset_correlation_id,
        set_correlation_id,
    )

    trade_id: str = args.trade_id
    token = set_correlation_id(trade_id)
    try:
        active_cid = get_correlation_id()
        _emit({"correlation_id": active_cid, "trade_id": trade_id})
        settings = _load_settings()
        entries = _query_audit_log(trade_id, settings.database_url)
        _emit_audit_entries(trade_id, active_cid, entries)
    finally:
        reset_correlation_id(token)


def _emit_audit_entries(
    trade_id: str,
    correlation_id: str,
    entries: list[dict[str, Any]],
) -> None:
    """Emit NDJSON audit entries, or a not-found record when *entries* is empty."""
    if not entries:
        _emit(
            {
                "status": "not_found",
                "message": (
                    f"No audit entries found for trade {trade_id}. "
                    "Ensure the trade exists and DATABASE_URL is correct."
                ),
            }
        )
        return
    for entry in entries:
        _emit(entry)
    _emit(
        {
            "status": "ok",
            "trade_id": trade_id,
            "correlation_id": correlation_id,
            "entry_count": len(entries),
        }
    )


def _fetch_audit_rows(url: str, trade_id: str) -> list[Any]:
    """Execute the audit_log SQL query and return raw dict rows."""
    import psycopg  # deferred: heavy import
    import psycopg.rows

    with psycopg.connect(url, row_factory=psycopg.rows.dict_row) as conn:  # pyright: ignore[reportArgumentType]
        return conn.execute(
            """
            SELECT
                al.id::text          AS id,
                al.correlation_id::text AS correlation_id,
                al.actor,
                al.action,
                al.payload,
                al.ts::text          AS ts
            FROM audit_log al
            JOIN trades t ON t.cycle_id = al.correlation_id
            WHERE t.id = %s::uuid
            ORDER BY al.ts
            """,
            (trade_id,),
        ).fetchall()


def _query_audit_log(trade_id: str, database_url: str) -> list[dict[str, Any]]:
    """Return audit entries for *trade_id*; empty list on any DB error."""
    try:
        from firm.orchestration.checkpointer import _normalise_database_url

        rows = _fetch_audit_rows(_normalise_database_url(database_url), trade_id)
        return [dict(row) for row in rows]
    except Exception:
        logger.warning("Audit-log query for trade %s failed", trade_id, exc_info=True)
        return []
