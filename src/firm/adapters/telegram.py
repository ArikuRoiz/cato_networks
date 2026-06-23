"""Telegram Bot adapter for the HITL approval surface.

Send/receive flow
-----------------
1. ``send_hitl_request`` POSTs ``sendMessage`` to the configured chat with an
   inline keyboard (Approve / Reject / Edit).
2. It then long-polls ``getUpdates`` (offset-based, 30-second windows) until a
   ``callback_query`` whose ``callback_data`` encodes the correlation-id arrives
   or the request expires.
3. The matching ``answerCallbackQuery`` clears the spinner and posts a
   confirmation message.
4. Returns ``ApprovalResult(status=APPROVED | REJECTED | EXPIRED)``.

Disabled / dry-run mode
-----------------------
When ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` are absent (or are
placeholder strings ending in ``...``), the adapter runs in dry-run mode: all
payloads are logged at INFO level, and ``send_hitl_request`` returns EXPIRED
(fail-safe — never auto-approve).

Token / chat_id can be supplied via constructor args or via the environment
variables ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from firm.domain.enums import ApprovalStatus
from firm.ports.types import ApprovalResult, HITLRequest

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_POLL_WINDOW_SECONDS = 30
_POLL_SLEEP_SECONDS = 1

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallbackUpdate:
    """Parsed ``callback_query`` from a Telegram getUpdates response."""

    update_id: int
    callback_query_id: str
    callback_data: str
    from_username: str | None


@dataclass(frozen=True)
class _PollResult:
    """Outcome of the long-poll loop."""

    matched: _CallbackUpdate | None

    @property
    def timed_out(self) -> bool:
        return self.matched is None


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class TelegramHITL:
    """Telegram Bot HITL adapter — implements the ``ReportSink`` surface.

    Parameters
    ----------
    token:
        Telegram Bot API token (``123456:ABC-...``).  Falls back to
        ``TELEGRAM_BOT_TOKEN`` env var; absent / placeholder → dry-run mode.
    chat_id:
        Telegram chat ID to post messages to.  Falls back to
        ``TELEGRAM_CHAT_ID`` env var; absent → dry-run mode.
    http_client:
        Optional pre-built ``httpx.Client`` for testing; a default
        synchronous client is constructed when omitted.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_token = token or os.getenv("TELEGRAM_BOT_TOKEN") or ""
        resolved_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID") or ""

        self._dry_run = not _is_real_credential(resolved_token) or not resolved_chat_id
        self._token = resolved_token
        self._chat_id = resolved_chat_id
        self._http = http_client or (httpx.Client() if not self._dry_run else None)

        if self._dry_run:
            logger.info(
                "TelegramHITL running in dry-run mode "
                "(TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID absent/placeholder). "
                "send_hitl_request will return EXPIRED."
            )

    # ------------------------------------------------------------------
    # ReportSink interface
    # ------------------------------------------------------------------

    def send_daily_report(self, report: object) -> None:
        """Log-only stub — daily reports are delivered via SlackReportSink."""
        logger.info("TelegramHITL: send_daily_report suppressed (Telegram is HITL-only).")

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Post the trade summary + inline keyboard; block-poll until decided.

        Returns EXPIRED immediately when:
          - dry-run mode (missing credentials)
          - the request has already expired before posting
          - the poll window closes without a matching callback

        Returns APPROVED or REJECTED when the operator taps a button.
        Never auto-approves on timeout (fail-safe).
        """
        if self._dry_run:
            return _dry_run_result(req)

        if _is_expired(req.expires_at):
            logger.warning(
                "TelegramHITL: request %s already expired before posting", req.correlation_id
            )
            return ApprovalResult(status=ApprovalStatus.EXPIRED)

        message_id = self._send_hitl_message(req)
        if message_id is None:
            return ApprovalResult(status=ApprovalStatus.EXPIRED)

        poll_result = self._poll_for_callback(req)
        if poll_result.timed_out:
            self._post_text("⏰ HITL request expired — trade auto-rejected (fail-safe).")
            return ApprovalResult(status=ApprovalStatus.EXPIRED)

        return self._finalise(poll_result.matched, req)  # type: ignore[arg-type]

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Post (or log) an operational alert."""
        text = f"⚠️ [{correlation_id}] {message}"
        if self._dry_run:
            logger.info("TelegramHITL dry-run: send_alert %s", text)
            return
        self._post_text(text)

    # ------------------------------------------------------------------
    # Private: Telegram API calls
    # ------------------------------------------------------------------

    def _send_hitl_message(self, req: HITLRequest) -> int | None:
        """POST sendMessage with inline keyboard; return message_id or None on error."""
        payload = _build_sendmessage_payload(self._chat_id, req)
        try:
            data = self._call("sendMessage", payload)
            result = data["result"]
            if not isinstance(result, dict):
                return None
            return int(result["message_id"])
        except Exception:
            logger.exception(
                "TelegramHITL: sendMessage failed for correlation_id=%s", req.correlation_id
            )
            return None

    def _poll_for_callback(self, req: HITLRequest) -> _PollResult:
        """Long-poll getUpdates until a matching callback_query or expiry."""
        offset = 0
        while not _is_expired(req.expires_at):
            updates = self._fetch_updates(offset)
            for update in updates:
                raw_id = update.get("update_id")
                if isinstance(raw_id, int):
                    offset = max(offset, raw_id + 1)
                parsed = _parse_callback_update(update)
                if parsed is None:
                    continue
                if _matches_request(parsed, req):
                    return _PollResult(matched=parsed)
            time.sleep(_POLL_SLEEP_SECONDS)
        return _PollResult(matched=None)

    def _fetch_updates(self, offset: int) -> list[dict[str, object]]:
        """Call getUpdates and return raw update dicts; empty list on any error."""
        try:
            data = self._call("getUpdates", {"offset": offset, "timeout": _POLL_WINDOW_SECONDS})
            raw = data.get("result")
            if not isinstance(raw, list):
                return []
            return [item for item in raw if isinstance(item, dict)]
        except Exception:
            logger.exception("TelegramHITL: getUpdates failed")
            return []

    def _finalise(self, callback: _CallbackUpdate, req: HITLRequest) -> ApprovalResult:
        """Answer the callback, post confirmation, and return the mapped result."""
        result = _map_callback_to_result(callback)
        confirmation = _confirmation_text(result.status, callback.from_username)
        self._answer_callback(callback.callback_query_id, confirmation)
        self._post_text(confirmation)
        return result

    def _answer_callback(self, callback_query_id: str, text: str) -> None:
        """Call answerCallbackQuery to clear the button spinner."""
        try:
            self._call("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})
        except Exception:
            logger.exception("TelegramHITL: answerCallbackQuery failed")

    def _post_text(self, text: str) -> None:
        """Post a plain-text follow-up message; swallow errors."""
        try:
            self._call("sendMessage", {"chat_id": self._chat_id, "text": text})
        except Exception:
            logger.exception("TelegramHITL: follow-up sendMessage failed")

    def _call(self, method: str, payload: dict[str, object]) -> dict[str, object]:
        """Execute one Telegram Bot API call and return the parsed JSON response."""
        url = _API_BASE.format(token=self._token, method=method)
        assert self._http is not None  # guaranteed when not in dry-run
        resp = self._http.post(url, json=payload, timeout=_POLL_WINDOW_SECONDS + 5)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data


# ---------------------------------------------------------------------------
# Validators / pure helpers
# ---------------------------------------------------------------------------


def _is_real_credential(token: str) -> bool:
    """Return True only for a plausibly real Telegram bot token.

    Filters out placeholder strings such as ``123456:ABC-...`` that live in
    .env templates and would cause live API calls to fail immediately.
    """
    return bool(token) and not token.endswith("...")


def _is_expired(expires_at: datetime) -> bool:
    if expires_at.tzinfo is None:
        return True
    return datetime.now(tz=UTC) >= expires_at


def _dry_run_result(req: HITLRequest) -> ApprovalResult:
    """Log the HITL request and return EXPIRED (fail-safe dry-run)."""
    logger.info(
        "TelegramHITL dry-run: send_hitl_request "
        "symbol=%s side=%s qty=%s notional=%s correlation_id=%s",
        req.symbol,
        req.side,
        req.qty_str,
        req.notional,
        req.correlation_id,
    )
    return ApprovalResult(status=ApprovalStatus.EXPIRED)


def _build_sendmessage_payload(chat_id: str, req: HITLRequest) -> dict[str, object]:
    """Build the sendMessage payload with trade summary + inline keyboard."""
    text = _format_hitl_text(req)
    keyboard = _inline_keyboard(req)
    return {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
        "reply_markup": {"inline_keyboard": keyboard},
    }


def _format_hitl_text(req: HITLRequest) -> str:
    """Format the trade summary in MarkdownV2 (Telegram escaping applied)."""
    notional_str = f"{req.notional:,.2f}"
    expires_str = req.expires_at.strftime("%H:%M:%S UTC")
    # MarkdownV2 requires escaping: . - ( ) = + { } ! # > |
    lines = [
        "🚨 *Trade Approval Required*",
        "",
        f"*Symbol:* `{req.symbol}`",
        f"*Side:* `{req.side.upper()}`",
        f"*Qty:* `{req.qty_str}`",
        f"*Notional:* `${_escape_mdv2(notional_str)}`",
        f"*Reason:* {_escape_mdv2(req.reason)}",
        f"*Expires:* {_escape_mdv2(expires_str)}",
        f"*ID:* `{req.correlation_id}`",
    ]
    return "\n".join(lines)


def _escape_mdv2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{ch}" if ch in special else ch for ch in text)


def _inline_keyboard(req: HITLRequest) -> list[list[dict[str, str]]]:
    """Build the two-row inline keyboard for Approve / Reject / Edit."""
    cid = req.correlation_id
    return [
        [
            {"text": "✅ Approve", "callback_data": f"approve:{cid}"},
            {"text": "❌ Reject", "callback_data": f"reject:{cid}"},
        ],
        [
            {"text": "✏️ Edit qty", "callback_data": f"edit:{cid}"},
        ],
    ]


def _parse_callback_update(raw: dict[str, object]) -> _CallbackUpdate | None:
    """Parse a raw update dict into a ``_CallbackUpdate``; return None if not a callback."""
    cq = raw.get("callback_query")
    if not isinstance(cq, dict):
        return None
    update_id = raw.get("update_id")
    callback_query_id = cq.get("id")
    data = cq.get("data")
    from_info = cq.get("from") or {}
    username = from_info.get("username") if isinstance(from_info, dict) else None
    if not isinstance(update_id, int) or not isinstance(callback_query_id, str) or not isinstance(data, str):
        return None
    return _CallbackUpdate(
        update_id=update_id,
        callback_query_id=callback_query_id,
        callback_data=data,
        from_username=str(username) if username else None,
    )


def _matches_request(callback: _CallbackUpdate, req: HITLRequest) -> bool:
    """Return True when the callback encodes a decision for *req*'s correlation_id."""
    parts = callback.callback_data.split(":", 1)
    if len(parts) != 2:
        return False
    _action, correlation_id = parts
    return correlation_id == req.correlation_id


def _map_callback_to_result(callback: _CallbackUpdate) -> ApprovalResult:
    """Map a callback_data action to an ``ApprovalResult``."""
    action = callback.callback_data.split(":", 1)[0]
    decided_by = callback.from_username or "telegram-user"
    if action == "approve":
        return ApprovalResult(status=ApprovalStatus.APPROVED, decided_by=decided_by)
    if action == "edit":
        # Edit taps are treated as APPROVED with no qty change in v1;
        # the operator can adjust the qty via a follow-up trade if needed.
        return ApprovalResult(status=ApprovalStatus.APPROVED, decided_by=decided_by)
    # "reject" or any unrecognised action → safe-default rejection
    return ApprovalResult(status=ApprovalStatus.REJECTED, decided_by=decided_by)


def _confirmation_text(status: ApprovalStatus, username: str | None) -> str:
    actor = username or "Risk Committee"
    if status == ApprovalStatus.APPROVED:
        return f"✅ Approved by {actor}"
    return f"❌ Rejected by {actor}"
