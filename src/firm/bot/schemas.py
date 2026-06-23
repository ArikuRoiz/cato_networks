"""Typed wrappers for inbound Telegram updates and bot state.

All external data (raw getUpdates dicts) is wrapped here at the boundary.
Downstream code consumes typed objects, never raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Parsed inbound update types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TextMessage:
    """A text message (command or bare ticker) from the operator."""

    update_id: int
    chat_id: int
    text: str

    @property
    def is_command(self) -> bool:
        return self.text.startswith("/")

    @property
    def command(self) -> str:
        """Command word without the leading slash, e.g. 'run'."""
        if not self.is_command:
            return ""
        return self.text.split()[0][1:].lower()

    @property
    def arg(self) -> str:
        """First whitespace-separated argument after the command word."""
        parts = self.text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return ""
        return parts[1].strip().split()[0].upper()

    @property
    def is_force(self) -> bool:
        """True when a 'force'/'--force' word follows the command and ticker.

        Mirrors ``firm run --force-buy``: ``/run NVDA force`` injects a
        high-conviction BUY so the HITL approval card always appears.
        """
        words = {w.lower() for w in self.text.split()[2:]}
        return bool(words & {"force", "--force"})


@dataclass(frozen=True)
class _CallbackTap:
    """An inline-keyboard button press from the operator.

    callback_data scheme (``:``-delimited)::

        approve:<cid>        — accept the recommended action
        reject:<cid>         — reject; show the alternatives keyboard
        act:<verb>:<cid>     — pick an alternative action (verb ∈ buy/sell/hold)

    The leading token is the *kind*; the trailing token is always the
    correlation_id.  For ``act`` taps the middle token is the resume *verb*
    handed straight to ``resume_decision``.
    """

    update_id: int
    callback_query_id: str
    chat_id: int
    callback_data: str
    from_username: str | None

    @property
    def _parts(self) -> list[str]:
        return self.callback_data.split(":")

    @property
    def kind(self) -> str:
        """Leading token: 'approve', 'reject', or 'act'."""
        return self._parts[0]

    @property
    def correlation_id(self) -> str:
        """Correlation ID is always the trailing token of callback_data."""
        parts = self._parts
        return parts[-1] if len(parts) >= 2 else ""

    @property
    def verb(self) -> str:
        """Resume verb for an 'act' tap (buy/sell/hold); '' otherwise."""
        parts = self._parts
        if self.kind == "act" and len(parts) == 3:
            return parts[1]
        return ""

    @property
    def is_approve(self) -> bool:
        return self.kind == "approve"

    @property
    def is_reject(self) -> bool:
        return self.kind == "reject"

    @property
    def is_action(self) -> bool:
        return self.kind == "act"


# ---------------------------------------------------------------------------
# In-flight run tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PendingRun:
    """Tracks a pipeline run that has paused at a HITL interrupt.

    Created when the graph emits an interrupt; consumed when the operator
    taps Approve or Reject in the chat.
    """

    correlation_id: str
    thread_id: str
    chat_id: int
    symbol: str
    recommendation: str


# ---------------------------------------------------------------------------
# Parse helpers (boundary wrapping)
# ---------------------------------------------------------------------------


def parse_text_message(raw: dict[str, Any]) -> _TextMessage | None:
    """Wrap a raw update dict into a _TextMessage; return None if not a text message."""
    msg = raw.get("message")
    if not isinstance(msg, dict):
        return None
    text = msg.get("text")
    chat = msg.get("chat")
    if not isinstance(text, str) or not isinstance(chat, dict):
        return None
    update_id = raw.get("update_id")
    chat_id = chat.get("id")
    if not isinstance(update_id, int) or not isinstance(chat_id, int):
        return None
    return _TextMessage(update_id=update_id, chat_id=chat_id, text=text.strip())


def parse_callback_tap(raw: dict[str, Any]) -> _CallbackTap | None:
    """Wrap a raw update dict into a _CallbackTap; return None if not a callback."""
    cq = raw.get("callback_query")
    if not isinstance(cq, dict):
        return None
    update_id = raw.get("update_id")
    cq_id = cq.get("id")
    data = cq.get("data")
    msg = cq.get("message") or {}
    chat = msg.get("chat") if isinstance(msg, dict) else None
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    from_info = cq.get("from") or {}
    username = from_info.get("username") if isinstance(from_info, dict) else None
    if (
        not isinstance(update_id, int)
        or not isinstance(cq_id, str)
        or not isinstance(data, str)
        or not isinstance(chat_id, int)
    ):
        return None
    return _CallbackTap(
        update_id=update_id,
        callback_query_id=cq_id,
        chat_id=chat_id,
        callback_data=data,
        from_username=str(username) if username else None,
    )
