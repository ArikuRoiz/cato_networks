"""Telegram bot service: single getUpdates consumer, non-blocking HITL.

Architecture
------------
One long-polling ``getUpdates`` loop (offset-based) routes all inbound
Telegram events:

  Text messages  → ``_handle_text``
    /run TICKER (or bare TICKER) → starts a live pipeline run in a background
    thread; the loop stays responsive to other commands while the run proceeds.
    /start, /help → greeting.
    /portfolio    → current NAV / cash / holdings.

  callback_query → ``_handle_callback``
    Approve/Reject tap → resumes the matching paused graph thread via
    ``Command(resume=..., update={"hitl_status": ...})``; the graph continues
    in the same background thread that is waiting on an Event.

HITL is non-blocking
--------------------
When a run hits the risk interrupt the background thread:
  1. Sends the rich approval card to the chat.
  2. Registers a ``_PendingRun`` in ``_pending`` (keyed by correlation_id).
  3. Waits on a threading.Event.

The main loop's callback handler finds the pending run, sets ``hitl_status``
and ``resume_value`` on the shared ``_ResumeSignal``, then fires the Event.
The background thread wakes up, reads the signal, and calls
``graph.stream(Command(resume=..., update=...))`` to continue execution.

There is exactly ONE ``getUpdates`` consumer.  ``TelegramHITL.send_hitl_request``
(the blocking poll) is never called from the bot service; only its message-
formatting helpers are reused.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from firm.bot.formatters import (
    format_approval_card,
    format_decision_report,
    format_portfolio_report,
)
from firm.bot.schemas import (
    _CallbackTap,
    _PendingRun,
    _TextMessage,
    parse_callback_tap,
    parse_text_message,
)
from firm.domain.enums import HITLStatus
from firm.orchestration.hitl import HITLDecision, parse_decision
from firm.ports.types import HITLRequest

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT_SECONDS = 30
_HITL_EXPIRY_MINUTES = 10

# ---------------------------------------------------------------------------
# Shared resume signal (background thread ↔ main loop)
# ---------------------------------------------------------------------------


@dataclass
class _ResumeSignal:
    """Carries the operator's decision from the callback handler to the run thread."""

    resume_value: str = ""
    hitl_status: str = ""
    event: threading.Event = field(default_factory=threading.Event)


# ---------------------------------------------------------------------------
# BotService
# ---------------------------------------------------------------------------


class BotService:
    """Single-consumer Telegram bot for operator-driven pipeline runs.

    Parameters
    ----------
    token:
        Telegram Bot API token.
    chat_id:
        The chat ID that is allowed to send commands and receives all messages.
    graph:
        A compiled LangGraph graph (with PostgresSaver checkpointer).
    settings:
        Application settings (used to load portfolio for /portfolio command).
    http_client:
        Optional pre-built ``httpx.Client`` for testing.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        graph: Any,
        settings: Any,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._graph = graph
        self._settings = settings
        self._http = http_client or httpx.Client()

        # correlation_id → (_PendingRun, _ResumeSignal)
        self._pending: dict[str, tuple[_PendingRun, _ResumeSignal]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block forever: poll getUpdates and dispatch to handlers."""
        logger.info("BotService starting; polling for updates…")
        offset = 0
        while True:
            updates = self._fetch_updates(offset)
            for raw in updates:
                raw_id = raw.get("update_id")
                if isinstance(raw_id, int):
                    offset = max(offset, raw_id + 1)
                self._dispatch(raw)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, raw: dict[str, Any]) -> None:
        tap = parse_callback_tap(raw)
        if tap is not None:
            self._handle_callback(tap)
            return
        msg = parse_text_message(raw)
        if msg is not None:
            self._handle_text(msg)

    def _handle_text(self, msg: _TextMessage) -> None:
        if msg.is_command:
            self._route_command(msg)
        else:
            # Bare text — treat as a ticker if it looks like one
            ticker = msg.text.upper().strip()
            if ticker.isalpha() and 1 <= len(ticker) <= 6:
                self._start_run(ticker, msg.chat_id)
            else:
                self._send_text(msg.chat_id, "Send /run TICKER or just a ticker symbol like NVDA.")

    def _route_command(self, msg: _TextMessage) -> None:
        cmd = msg.command
        if cmd == "run":
            ticker = msg.arg or ""
            if not ticker:
                self._send_text(msg.chat_id, "Usage: /run TICKER  (e.g. /run NVDA)")
                return
            self._start_run(ticker, msg.chat_id, force_buy=msg.is_force)
        elif cmd == "demo":
            ticker = msg.arg or ""
            if not ticker:
                self._send_text(msg.chat_id, "Usage: /demo TICKER  (e.g. /demo NVDA)")
                return
            self._start_run(ticker, msg.chat_id, force_buy=True)
        elif cmd in {"start", "help"}:
            self._send_help(msg.chat_id)
        elif cmd == "portfolio":
            self._send_portfolio(msg.chat_id)
        else:
            self._send_text(msg.chat_id, f"Unknown command /{cmd}. Send /help for usage.")

    def _handle_callback(self, tap: _CallbackTap) -> None:
        cid = tap.correlation_id
        with self._lock:
            entry = self._pending.get(cid)
        if entry is None:
            self._answer_callback(
                tap.callback_query_id, "No matching run found — may have expired."
            )
            return
        run, signal = entry

        # Reject is NOT a terminal decision: it does not wake the run thread.
        # Instead it offers the OTHER actions; the run keeps waiting on the
        # event until one of those alternatives is tapped.
        if tap.is_reject:
            self._offer_alternatives(tap, run)
            return

        decision = _decision_for(tap, run.recommendation)
        signal.resume_value = decision.value
        signal.hitl_status = decision.hitl_status
        signal.event.set()
        actor = tap.from_username or "operator"
        self._answer_callback(tap.callback_query_id, f"{_ack_verb(decision)} by {actor}")

    def _offer_alternatives(self, tap: _CallbackTap, run: _PendingRun) -> None:
        """Reject tap → send a follow-up keyboard with the other actions."""
        self._answer_callback(tap.callback_query_id, "Rejected — pick another action.")
        keyboard = _alternatives_keyboard(run.recommendation, run.correlation_id)
        self._send_message_with_keyboard(
            tap.chat_id,
            f"❌ *Rejected {run.recommendation.upper()} for {run.symbol}.* "
            "Choose an alternative action:",
            keyboard,
        )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def _start_run(self, ticker: str, chat_id: int, force_buy: bool = False) -> None:
        prefix = "🎬 Demo run (forced trade) for" if force_buy else "🔎 Researching"
        self._send_text(chat_id, f"{prefix} {ticker}…")
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(ticker, chat_id),
            kwargs={"force_buy": force_buy},
            daemon=True,
            name=f"pipeline-{ticker}",
        )
        thread.start()

    def _run_pipeline(self, symbol: str, chat_id: int, force_buy: bool = False) -> None:
        """Execute the graph in a background thread; handle HITL pause if it occurs."""
        correlation_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        initial_state: dict[str, Any] = {
            "symbol": symbol,
            "decision_ts": datetime.now(tz=UTC).isoformat(),
            "correlation_id": correlation_id,
            "force_buy": force_buy,
        }
        try:
            final_state = self._stream_to_interrupt(initial_state, config)
            interrupt_payload = self._detect_interrupt(config)

            if interrupt_payload is None:
                self._send_cycle_report(chat_id, symbol, final_state)
                return

            # Paused at HITL — send card, wait for operator tap
            self._send_hitl_card(chat_id, interrupt_payload, correlation_id, thread_id, symbol)
            signal = self._wait_for_decision(correlation_id)

            if signal is None:
                self._send_text(
                    chat_id, f"⏰ HITL expired for {symbol} — trade rejected (fail-safe)."
                )
                return

            recommendation = self._pending_recommendation(correlation_id)
            final_state = self._resume_graph(thread_id, signal)
            self._send_decision_report(
                chat_id, symbol, signal.hitl_status, final_state, recommendation
            )
            self._send_cycle_report(chat_id, symbol, final_state)

        except Exception:
            logger.exception("Pipeline error for %s", symbol)
            self._send_text(chat_id, f"⚠️ Pipeline error for {symbol}. Check logs.")
        finally:
            with self._lock:
                self._pending.pop(correlation_id, None)

    def _stream_to_interrupt(
        self,
        initial_state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Stream the graph until it pauses or finishes; return last state."""
        final_state: dict[str, Any] = {}
        for event in self._graph.stream(initial_state, config=config, stream_mode="values"):
            final_state = event
        return final_state

    def _detect_interrupt(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Return interrupt payload if the graph is paused, else None."""
        run_state = self._graph.get_state(config)
        if not (run_state.next and run_state.tasks):
            return None
        for task in run_state.tasks:
            if getattr(task, "interrupts", None):
                value = task.interrupts[0].value if task.interrupts else {}
                return value if isinstance(value, dict) else {}
        return None

    def _send_hitl_card(
        self,
        chat_id: int,
        interrupt_payload: dict[str, Any],
        correlation_id: str,
        thread_id: str,
        symbol: str,
    ) -> None:
        """Build and send the approval card; register the pending run."""
        req = _build_hitl_request(interrupt_payload, correlation_id)
        card_text = format_approval_card(req.model_dump())
        keyboard = _approval_keyboard(correlation_id)
        self._send_message_with_keyboard(chat_id, card_text, keyboard)

        pending = _PendingRun(
            correlation_id=correlation_id,
            thread_id=thread_id,
            chat_id=chat_id,
            symbol=symbol,
            recommendation=_normalize_recommendation(req.recommendation, req.side),
        )
        signal = _ResumeSignal()
        with self._lock:
            self._pending[correlation_id] = (pending, signal)

    def _wait_for_decision(self, correlation_id: str) -> _ResumeSignal | None:
        """Block until operator taps Approve/Reject or HITL expires."""
        with self._lock:
            entry = self._pending.get(correlation_id)
        if entry is None:
            return None
        _run, signal = entry
        deadline = _HITL_EXPIRY_MINUTES * 60
        fired = signal.event.wait(timeout=deadline)
        return signal if fired else None

    def _resume_graph(
        self,
        thread_id: str,
        signal: _ResumeSignal,
    ) -> dict[str, Any]:
        """Resume the interrupted graph via the shared HITL entry point."""
        from firm.orchestration.hitl import resume_decision

        return resume_decision(self._graph, thread_id, signal.resume_value)

    def _pending_recommendation(self, correlation_id: str) -> str | None:
        """Return the desk's recommendation for the pending run, if still registered."""
        with self._lock:
            entry = self._pending.get(correlation_id)
        return entry[0].recommendation if entry is not None else None

    def _send_decision_report(
        self,
        chat_id: int,
        symbol: str,
        hitl_status: str,
        final_state: dict[str, Any],
        recommendation: str | None = None,
    ) -> None:
        """Send the 'what I did & why' follow-up after approve/reject/override.

        ``hitl_decision`` (threaded through graph state by the risk node) tells the
        formatter whether this was a plain approve or an override-buy/sell/hold, so
        the label matches the action that actually executed.
        """
        report = format_decision_report(
            symbol=symbol,
            hitl_status=hitl_status,
            cycle_outcome=final_state.get("cycle_outcome"),
            synthesis=final_state.get("synthesis"),
            verdict=final_state.get("verdict"),
            approved_trade=final_state.get("approved_trade"),
            rejection_reason=_extract_rejection_reason(hitl_status, final_state),
            hitl_decision=final_state.get("hitl_decision"),
            recommendation=recommendation,
        )
        self._send_text(chat_id, report)

    def _send_cycle_report(
        self,
        chat_id: int,
        symbol: str,
        final_state: dict[str, Any],
    ) -> None:
        """Send the end-of-run report with NAV, outcome, memo."""
        outcome = final_state.get("cycle_outcome", "unknown")
        synthesis = final_state.get("synthesis") or {}
        verdict = final_state.get("verdict") or {}

        lines = [
            f"📋 *Run complete — {symbol}*",
            f"Outcome: *{outcome}*",
        ]
        title = synthesis.get("title")
        if title:
            lines.append(f"Memo: _{title}_")
        score = verdict.get("coherence_score")
        alignment = verdict.get("alignment")
        if score is not None:
            lines.append(f"Judge: {score}/5 {alignment or ''}")

        self._send_text(chat_id, "\n".join(lines))

    # ------------------------------------------------------------------
    # /portfolio command
    # ------------------------------------------------------------------

    def _send_portfolio(self, chat_id: int) -> None:
        try:
            snapshot = _load_portfolio_snapshot(self._settings)
            text = format_portfolio_report(
                nav=snapshot["nav"],
                cash=snapshot["cash"],
                pnl=snapshot["pnl"],
                holdings=snapshot["holdings"],
            )
            self._send_text(chat_id, text)
        except Exception:
            logger.exception("Failed to load portfolio")
            self._send_text(chat_id, "⚠️ Could not load portfolio. Is Postgres running?")

    # ------------------------------------------------------------------
    # /start, /help
    # ------------------------------------------------------------------

    def _send_help(self, chat_id: int) -> None:
        text = (
            "🤖 *AI Investment Firm Bot*\n\n"
            "Commands:\n"
            "  /run TICKER — research a stock and ask for trade approval\n"
            "  /run TICKER force — force a high-conviction BUY so the approval card always appears (demo)\n"
            "  /demo TICKER — same as /run TICKER force\n"
            "  /portfolio  — show current NAV, cash, and holdings\n"
            "  /help       — show this message\n\n"
            "You can also send a bare ticker: just type *NVDA*"
        )
        self._send_text(chat_id, text)

    # ------------------------------------------------------------------
    # Telegram API calls
    # ------------------------------------------------------------------

    def _fetch_updates(self, offset: int) -> list[dict[str, Any]]:
        try:
            data = self._call("getUpdates", {"offset": offset, "timeout": _POLL_TIMEOUT_SECONDS})
            raw = data.get("result")
            if not isinstance(raw, list):
                return []
            return [item for item in raw if isinstance(item, dict)]
        except Exception:
            logger.exception("getUpdates failed")
            return []

    def _send_text(self, chat_id: int, text: str) -> None:
        try:
            self._call("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        except Exception:
            logger.exception("sendMessage failed")

    def _send_message_with_keyboard(
        self,
        chat_id: int,
        text: str,
        keyboard: list[list[dict[str, str]]],
    ) -> None:
        try:
            self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "reply_markup": {"inline_keyboard": keyboard},
                },
            )
        except Exception:
            logger.exception("sendMessage (with keyboard) failed")

    def _answer_callback(self, callback_query_id: str, text: str) -> None:
        try:
            self._call(
                "answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text}
            )
        except Exception:
            logger.exception("answerCallbackQuery failed")

    def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = _API_BASE.format(token=self._token, method=method)
        resp = self._http.post(url, json=payload, timeout=_POLL_TIMEOUT_SECONDS + 5)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


# Recommendation verb → its two alternative action verbs (label, callback verb).
_ALTERNATIVES: dict[str, list[str]] = {
    "buy": ["hold", "sell"],
    "hold": ["buy", "sell"],
    "sell": ["buy", "hold"],
}
_ACTION_EMOJI: dict[str, str] = {"buy": "🟢", "sell": "🔴", "hold": "⏸️"}


def _decision_for(tap: _CallbackTap, recommendation: str) -> HITLDecision:
    """Resolve a callback tap to a structured HITL decision.

    Approve → accept the recommended action; an 'act' tap → the chosen verb.
    """
    if tap.is_approve:
        return HITLDecision.APPROVE
    return parse_decision(tap.verb or recommendation)


def _ack_verb(decision: HITLDecision) -> str:
    """Short Telegram toast label for a resolved decision."""
    labels = {
        HITLDecision.APPROVE: "✅ Approved",
        HITLDecision.OVERRIDE_BUY: "🟢 Buy",
        HITLDecision.OVERRIDE_SELL: "🔴 Sell",
        HITLDecision.OVERRIDE_HOLD: "⏸️ Hold",
    }
    return labels.get(decision, "Decided")


def _normalize_recommendation(recommendation: str | None, side: str) -> str:
    """Coerce the research recommendation to a buy/sell/hold verb.

    Falls back to the trade *side* (the proposal is what risk sized), then buy.
    """
    candidate = (recommendation or side or "buy").strip().lower()
    if "sell" in candidate or "reduce" in candidate or "trim" in candidate:
        return "sell"
    if "hold" in candidate or "neutral" in candidate:
        return "hold"
    return "buy"


def _approval_keyboard(correlation_id: str) -> list[list[dict[str, str]]]:
    return [
        [
            {"text": "✅ Approve", "callback_data": f"approve:{correlation_id}"},
            {"text": "❌ Reject", "callback_data": f"reject:{correlation_id}"},
        ],
    ]


def _alternatives_keyboard(recommendation: str, correlation_id: str) -> list[list[dict[str, str]]]:
    """Buttons for the OTHER actions after a Reject tap.

    callback_data is ``act:<verb>:<cid>`` — verb is fed to ``resume_decision``.
    """
    verbs = _ALTERNATIVES.get(recommendation, _ALTERNATIVES["buy"])
    return [
        [
            {
                "text": f"{_ACTION_EMOJI.get(verb, '')} {verb.title()}",
                "callback_data": f"act:{verb}:{correlation_id}",
            }
            for verb in verbs
        ]
    ]


def _build_hitl_request(interrupt_payload: dict[str, Any], correlation_id: str) -> HITLRequest:
    """Construct a typed HITLRequest from the graph's interrupt payload."""
    return HITLRequest.from_interrupt(
        interrupt_payload, correlation_id, expiry_minutes=_HITL_EXPIRY_MINUTES
    )


def _extract_rejection_reason(hitl_status: str, final_state: dict[str, Any]) -> str | None:
    if hitl_status == HITLStatus.REJECTED:
        return "operator rejected"
    if hitl_status == HITLStatus.EXPIRED:
        return "HITL request timed out"
    return final_state.get("error")


def _load_portfolio_snapshot(settings: Any) -> dict[str, Any]:
    """Load current portfolio from Postgres and return a summary dict."""
    from sqlalchemy import create_engine

    from firm.adapters.market_data_live import LiveMarketData
    from firm.persistence.db_url import to_sqlalchemy_url
    from firm.persistence.ledger import FIRM_PORTFOLIO_ID, LedgerRepository

    engine = create_engine(to_sqlalchemy_url(settings.database_url))
    ledger = LedgerRepository(engine)
    portfolio = ledger.get_portfolio(FIRM_PORTFOLIO_ID)
    symbols = list(portfolio.holdings.keys())
    prices: dict[str, Decimal] = {}
    if symbols:
        md = LiveMarketData()
        now = datetime.now(tz=UTC)
        for sym in symbols:
            try:
                bar = md.get_bar(sym, now)
                if bar is not None:
                    prices[sym] = Decimal(str(bar.close))
            except Exception:
                logger.warning("Failed to fetch price for %s; omitting from NAV", sym, exc_info=True)
    nav = portfolio.nav(prices)
    cash = portfolio.cash
    pnl = nav - cash  # simplified: NAV - cash = unrealised P&L contribution
    holdings = [
        {"symbol": sym, "qty": str(h.quantity)}
        for sym, h in portfolio.holdings.items()
        if h.quantity > 0
    ]
    return {
        "nav": f"{nav:,.2f}",
        "cash": f"{cash:,.2f}",
        "pnl": f"{pnl:+,.2f}",
        "holdings": holdings,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_bot_service(
    settings: Any, graph: Any, http_client: httpx.Client | None = None
) -> BotService:
    """Construct a ``BotService`` from application settings and a compiled graph."""
    return BotService(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        graph=graph,
        settings=settings,
        http_client=http_client,
    )
