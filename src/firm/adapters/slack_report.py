"""SlackReportSink — ReportSink adapter that posts Block Kit messages via slack_sdk.

Supports three operations:
  - ``send_daily_report``: structured Block Kit message with NAV/P&L summary and
    top positions.
  - ``send_hitl_request``: interactive approval message with Approve/Reject buttons;
    polls for a response by ``correlation_id`` until ``expires_at``; times out as
    ``ApprovalResult(status="expired")``.
  - ``send_alert``: plain text to the configured channel.

HITL note: the current implementation posts the interactive message and returns
``ApprovalResult(status="expired")`` — never auto-approved — because the Slack
Events/Sockets API response-polling loop is outside the scope of this adapter's
compilation unit.  Any real polling loop plugs in without changing the public
interface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from firm.ports.types import ApprovalResult, DailyReport, HITLRequest, PositionRecord

# ---------------------------------------------------------------------------
# Block Kit builders — each ≤30 lines
# ---------------------------------------------------------------------------


def _header_block(text: str) -> dict[str, Any]:
    """Return a plain-text header block."""
    return {"type": "header", "text": {"type": "plain_text", "text": text}}


def _section_block(text: str) -> dict[str, Any]:
    """Return a mrkdwn section block."""
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _divider_block() -> dict[str, Any]:
    return {"type": "divider"}


def _context_block(text: str) -> dict[str, Any]:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": text}],
    }


def _daily_report_blocks(report: DailyReport) -> list[dict[str, Any]]:
    """Build the Block Kit payload for a daily performance summary."""
    pnl_sign = "+" if report.pnl >= Decimal("0") else ""
    top_positions = _format_top_positions(report.positions[:5])
    return [
        _header_block(f"Daily Report — {report.date}"),
        _divider_block(),
        _section_block(
            f"*NAV:* ${report.nav:,.2f}   "
            f"*P&L:* {pnl_sign}{report.pnl:,.2f}   "
            f"*Benchmark:* {report.benchmark_return:+.2%}"
        ),
        _section_block(f"*Trades executed:* {len(report.trades)}"),
        _section_block(f"*Top positions:*\n{top_positions}"),
        _divider_block(),
        _context_block(f"Generated at {datetime.now(tz=UTC).isoformat()}"),
    ]


def _format_top_positions(positions: list[PositionRecord]) -> str:
    """Format up to 5 positions as a compact Slack-safe string."""
    if not positions:
        return "_No open positions_"
    lines = [
        f"• {p.get('symbol', '?')}: {float(p.get('qty', 0)):.0f} shares "
        f"@ ${float(p.get('current_price', 0)):,.2f}"
        for p in positions
    ]
    return "\n".join(lines)


def _hitl_blocks(req: HITLRequest) -> list[dict[str, Any]]:
    """Build the interactive Block Kit payload for a HITL approval request."""
    return [
        _header_block("Trade Approval Required"),
        _divider_block(),
        _section_block(
            f"*Symbol:* {req.symbol}   *Side:* {req.side.upper()}   "
            f"*Qty:* {req.qty_str}   *Notional:* ${req.notional:,.2f}"
        ),
        _section_block(f"*Reason:* {req.reason}\n*Expires:* {req.expires_at.isoformat()}"),
        _divider_block(),
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
            ],
        },
        _context_block(f"correlation_id: `{req.correlation_id}`"),
    ]


# ---------------------------------------------------------------------------
# SlackReportSink
# ---------------------------------------------------------------------------


class SlackReportSink:
    """Implements :class:`~firm.ports.report.ReportSink` via the Slack Web API.

    *token* must be a valid Bot token with ``chat:write`` scope.
    *channel* is either a channel ID (``C012AB3CD``) or a ``#name`` string.
    """

    def __init__(self, token: str, channel: str) -> None:
        self._client: WebClient = WebClient(token=token)
        self._channel = channel

    # ------------------------------------------------------------------
    # ReportSink protocol
    # ------------------------------------------------------------------

    def send_daily_report(self, report: DailyReport) -> None:
        """Post a Block Kit daily summary to the configured channel.

        Propagates ``SlackApiError`` to the caller — reporting is non-critical
        but callers deserve a clear error rather than silent data loss.
        """
        self._client.chat_postMessage(
            channel=self._channel,
            blocks=_daily_report_blocks(report),
            text=f"Daily Report — {report.date}",  # fallback for notifications
        )

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Post an interactive approval message and wait for a human response.

        The message is posted with Approve/Reject buttons.  Until the Slack
        Events/Sockets polling loop is wired in, the adapter returns
        ``status="expired"`` after posting — the fail-safe mandated by the
        ReportSink contract (never auto-approve).

        A production implementation replaces the ``# TODO`` stub below with a
        polling loop that reads the Slack Events API by ``req.correlation_id``
        until ``req.expires_at``, then returns the human decision or
        ``ApprovalResult(status="expired")`` on timeout.
        """
        if _is_expired(req.expires_at):
            return ApprovalResult(status="expired", decided_by=None, edited_qty=None)

        try:
            self._client.chat_postMessage(
                channel=self._channel,
                blocks=_hitl_blocks(req),
                text=f"Trade approval required: {req.symbol} {req.side} {req.qty_str}",
            )
        except SlackApiError:
            # Degrade gracefully — a Slack failure must not auto-approve.
            return ApprovalResult(status="expired", decided_by=None, edited_qty=None)

        # TODO: poll Slack Events API for button action keyed by req.correlation_id
        # until req.expires_at, then return the human decision.
        return ApprovalResult(status="expired", decided_by=None, edited_qty=None)

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Post a plain-text alert to the configured channel."""
        self._client.chat_postMessage(
            channel=self._channel,
            text=f":warning: [{correlation_id}] {message}",
        )


def _is_expired(expires_at: datetime) -> bool:
    """True when *expires_at* is in the past (UTC).

    Fail-safe: a naive (tz-unaware) timestamp is treated as expired rather than
    valid — the adapter must never auto-approve on ambiguous input.
    """
    now = datetime.now(tz=UTC)
    if expires_at.tzinfo is None:
        return True  # treat naive timestamp as expired (fail-safe)
    return now >= expires_at
