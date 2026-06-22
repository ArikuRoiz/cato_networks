from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from firm.domain.enums import ApprovalStatus
from firm.ports.types import ApprovalResult, DailyReport, HITLRequest, PositionRecord


class SlackReportSink:
    def __init__(self, token: str, channel: str) -> None:
        self._client: WebClient = WebClient(token=token)
        self._channel = channel

    def send_daily_report(self, report: DailyReport) -> None:
        self._client.chat_postMessage(
            channel=self._channel,
            blocks=_daily_report_blocks(report),
            text=f"Daily Report — {report.date}",
        )

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        if _is_expired(req.expires_at):
            return ApprovalResult(status=ApprovalStatus.EXPIRED)
        try:
            self._client.chat_postMessage(
                channel=self._channel,
                blocks=_hitl_blocks(req),
                text=f"Trade approval required: {req.symbol} {req.side} {req.qty_str}",
            )
        except SlackApiError:
            return ApprovalResult(status=ApprovalStatus.EXPIRED)
        # TODO: poll Slack Events API for button action keyed by req.correlation_id
        return ApprovalResult(status=ApprovalStatus.EXPIRED)

    def send_alert(self, message: str, correlation_id: str) -> None:
        self._client.chat_postMessage(
            channel=self._channel,
            text=f":warning: [{correlation_id}] {message}",
        )


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
                {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "style": "primary", "value": "approved", "action_id": "hitl_approve"},
                {"type": "button", "text": {"type": "plain_text", "text": "Reject"}, "style": "danger", "value": "rejected", "action_id": "hitl_reject"},
            ],
        },
        _context(f"correlation_id: `{req.correlation_id}`"),
    ]


def _format_positions(positions: list[PositionRecord]) -> str:
    if not positions:
        return "_No open positions_"
    return "\n".join(
        f"• {p['symbol']}: {p['qty']:.0f} shares @ ${p['current_price']:,.2f}"
        for p in positions
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
