"""Slack adapter for ReportSink — live mode (token present) and dry-run (no token).

Dry-run behaviour
-----------------
When constructed without a ``token`` (or with an empty string), the sink never
calls the Slack API.  Instead it logs the Block Kit payload at INFO level so the
operator can inspect the message without a live Slack workspace.  ``send_hitl_request``
returns ``ApprovalResult(status=PENDING)`` in both modes — the real decision gate is
the LangGraph interrupt; this sink is only the *surface* that presents the request.

Token / channel can also be supplied via the environment variables
``SLACK_BOT_TOKEN`` and ``SLACK_CHANNEL`` as a convenience.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from firm.domain.enums import ApprovalStatus
from firm.ports.types import ApprovalResult, DailyReport, HITLRequest, PositionRecord

logger = logging.getLogger(__name__)


class SlackReportSink:
    """Slack report sink with graceful offline degradation.

    Parameters
    ----------
    token:
        Slack Bot token (``xoxb-...``).  When absent or empty the sink runs in
        dry-run mode: all payloads are logged rather than posted.
    channel:
        Slack channel name or ID (e.g. ``#trading-desk``).
    """

    def __init__(
        self,
        token: str | None = None,
        channel: str = "#trading-desk",
    ) -> None:
        resolved_token: str = token or os.getenv("SLACK_BOT_TOKEN") or ""
        self._channel = channel or os.getenv("SLACK_CHANNEL", "#trading-desk")
        self._dry_run = not _is_real_token(resolved_token)
        self._client = _build_client(resolved_token) if not self._dry_run else None

    # ------------------------------------------------------------------
    # ReportSink interface
    # ------------------------------------------------------------------

    def send_daily_report(self, report: DailyReport) -> None:
        """Post (or log) the daily performance summary."""
        blocks = _daily_report_blocks(report)
        text = f"Daily Report — {report.date}"
        if self._dry_run:
            logger.info(
                "SlackReportSink dry-run: send_daily_report payload\n%s",
                json.dumps({"text": text, "blocks": blocks}, indent=2),
            )
            return
        self._client.chat_postMessage(channel=self._channel, blocks=blocks, text=text)  # type: ignore[union-attr]

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Post (or log) the HITL approval request and return PENDING.

        Returns EXPIRED immediately if the request has already timed out.
        On SlackApiError the message was never delivered — return EXPIRED.
        Returns PENDING in all success paths (dry-run or live), because the
        real approval gate is the LangGraph interrupt, not this adapter.
        """
        if _is_expired(req.expires_at):
            return ApprovalResult(status=ApprovalStatus.EXPIRED)

        blocks = _hitl_blocks(req)
        text = f"Trade approval required: {req.symbol} {req.side} {req.qty_str}"

        if self._dry_run:
            logger.info(
                "SlackReportSink dry-run: send_hitl_request payload\n%s",
                json.dumps({"text": text, "blocks": blocks}, indent=2, default=str),
            )
            return ApprovalResult(status=ApprovalStatus.PENDING)

        try:
            self._client.chat_postMessage(channel=self._channel, blocks=blocks, text=text)  # type: ignore[union-attr]
        except SlackApiError:
            logger.exception("SlackReportSink: failed to post HITL request %s", req.correlation_id)
            return ApprovalResult(status=ApprovalStatus.EXPIRED)

        # Message delivered — gate is open; LangGraph interrupt handles the wait.
        return ApprovalResult(status=ApprovalStatus.PENDING)

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Post (or log) an operational alert."""
        text = f":warning: [{correlation_id}] {message}"
        if self._dry_run:
            logger.info("SlackReportSink dry-run: send_alert %s", text)
            return
        self._client.chat_postMessage(channel=self._channel, text=text)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_real_token(token: str) -> bool:
    """Return True only for a plausibly real Slack bot token.

    Filters out placeholder strings such as ``xoxb-...`` that live in .env
    templates and would cause live API calls to fail immediately.
    """
    return bool(token) and not token.endswith("...")


def _build_client(token: str) -> WebClient:
    """Return a ``WebClient`` for *token* (caller guarantees it is real)."""
    return WebClient(token=token)


# ---------------------------------------------------------------------------
# Block Kit builders
# ---------------------------------------------------------------------------


def _daily_report_blocks(report: DailyReport) -> list[dict[str, Any]]:
    pnl_sign = "+" if report.pnl >= Decimal("0") else ""
    return [
        _header(f"Daily Report — {report.date}"),
        _divider(),
        _section(
            f"*NAV:* ${report.nav:,.2f}   "
            f"*P&L:* {pnl_sign}{report.pnl:,.2f}   "
            f"*Benchmark:* {report.benchmark_return:+.2%}"
        ),
        _section(f"*Trades executed:* {len(report.trades)}"),
        _section(f"*Top positions:*\n{_format_positions(report.positions[:5])}"),
        _divider(),
        _context(f"Generated at {datetime.now(tz=UTC).isoformat()}"),
    ]


def _hitl_blocks(req: HITLRequest) -> list[dict[str, Any]]:
    """Build a Block Kit HITL approval message with Approve / Reject / Edit buttons."""
    return [
        _header("Trade Approval Required"),
        _divider(),
        _section(
            f"*Symbol:* {req.symbol}   *Side:* {req.side.upper()}   "
            f"*Qty:* {req.qty_str}   *Notional:* ${req.notional:,.2f}"
        ),
        _section(f"*Reason:* {req.reason}\n*Expires:* {req.expires_at.isoformat()}"),
        _divider(),
        {
            "type": "actions",
            "block_id": f"hitl_{req.correlation_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "style": "primary",
                    "value": "approved",
                    "action_id": "hitl_approve",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "value": "rejected",
                    "action_id": "hitl_reject",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Edit qty"},
                    "value": "edit",
                    "action_id": "hitl_edit",
                },
            ],
        },
        _context(f"correlation_id: `{req.correlation_id}`  |  trade_id: `{req.trade_id}`"),
    ]


def _format_positions(positions: list[PositionRecord]) -> str:
    if not positions:
        return "_No open positions_"
    return "\n".join(
        f"• {p['symbol']}: {p['qty']:.0f} shares @ ${p['current_price']:,.2f}" for p in positions
    )


def _header(text: str) -> dict[str, Any]:
    return {"type": "header", "text": {"type": "plain_text", "text": text}}


def _section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _divider() -> dict[str, Any]:
    return {"type": "divider"}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _is_expired(expires_at: datetime) -> bool:
    if expires_at.tzinfo is None:
        return True
    return datetime.now(tz=UTC) >= expires_at
