"""Unit tests for the Telegram bot service.

All tests use mocked httpx.Client and a mocked graph — no live network,
no live Postgres, no live LangGraph execution.

Coverage
--------
- Command routing: /run TICKER, bare ticker, /help, /start, /portfolio, /demo, unknown
- Callback → resume dispatch (approve → resume approve; reject → alternatives
  keyboard, run keeps waiting; act:<verb>:<cid> → resume that verb; unknown CID)
- Rich-card formatter (research_plan → human pros/cons; Hold → "no trade to place")
- Post-decision report builder (approve+fill, override-buy/sell/hold, rejected)
- Portfolio report builder
- parse_text_message / parse_callback_tap boundary wrappers
- _decision_for / _build_hitl_request / _approval_keyboard / _alternatives_keyboard
- CLI 'bot' subcommand is registered in the parser
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
from firm.bot.service import (
    BotService,
    _alternatives_keyboard,
    _approval_keyboard,
    _build_hitl_request,
    _decision_for,
    _normalize_recommendation,
    _ResumeSignal,
)
from firm.domain.enums import HITLStatus
from firm.orchestration.hitl import HITLDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot(graph: Any = None, http: Any = None) -> BotService:
    settings = MagicMock()
    settings.telegram_bot_token = "123:REAL"
    settings.telegram_chat_id = "-100123"
    settings.database_url = "postgresql://firm:firm@localhost:5432/firm"
    return BotService(
        token="123:REAL",
        chat_id="-100123",
        graph=graph or MagicMock(),
        settings=settings,
        http_client=http or MagicMock(),
    )


def _raw_message(text: str, chat_id: int = 123) -> dict[str, Any]:
    return {
        "update_id": 1,
        "message": {"text": text, "chat": {"id": chat_id}},
    }


def _raw_callback(data: str, chat_id: int = 123, update_id: int = 2) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cq-1",
            "data": data,
            "from": {"username": "risk_officer"},
            "message": {"chat": {"id": chat_id}},
        },
    }


def _pending(cid: str, recommendation: str = "buy") -> _PendingRun:
    return _PendingRun(
        correlation_id=cid,
        thread_id="tid-1",
        chat_id=123,
        symbol="NVDA",
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# parse_text_message
# ---------------------------------------------------------------------------


class TestParseTextMessage:
    def test_parses_command(self) -> None:
        msg = parse_text_message(_raw_message("/run NVDA"))
        assert msg is not None
        assert msg.is_command
        assert msg.command == "run"
        assert msg.arg == "NVDA"

    def test_parses_bare_text(self) -> None:
        msg = parse_text_message(_raw_message("hello world"))
        assert msg is not None
        assert not msg.is_command

    def test_returns_none_for_callback(self) -> None:
        assert parse_text_message(_raw_callback("approve:cid-123")) is None

    def test_returns_none_for_missing_text(self) -> None:
        raw = {"update_id": 1, "message": {"chat": {"id": 1}}}
        assert parse_text_message(raw) is None


# ---------------------------------------------------------------------------
# parse_callback_tap
# ---------------------------------------------------------------------------


class TestParseCallbackTap:
    def test_parses_approve(self) -> None:
        cid = str(uuid.uuid4())
        tap = parse_callback_tap(_raw_callback(f"approve:{cid}"))
        assert tap is not None
        assert tap.is_approve
        assert tap.correlation_id == cid
        assert tap.from_username == "risk_officer"

    def test_parses_reject(self) -> None:
        cid = str(uuid.uuid4())
        tap = parse_callback_tap(_raw_callback(f"reject:{cid}"))
        assert tap is not None
        assert tap.is_reject

    def test_parses_act_verb(self) -> None:
        cid = str(uuid.uuid4())
        tap = parse_callback_tap(_raw_callback(f"act:sell:{cid}"))
        assert tap is not None
        assert tap.is_action
        assert tap.verb == "sell"
        assert tap.correlation_id == cid

    def test_returns_none_for_text_message(self) -> None:
        assert parse_callback_tap(_raw_message("/run NVDA")) is None

    def test_returns_none_when_data_missing(self) -> None:
        raw = {
            "update_id": 1,
            "callback_query": {"id": "cq-1", "from": {}, "message": {"chat": {"id": 1}}},
        }
        assert parse_callback_tap(raw) is None


# ---------------------------------------------------------------------------
# _TextMessage predicates
# ---------------------------------------------------------------------------


class TestTextMessagePredicates:
    def test_run_command_extracts_arg(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="/run NVDA")
        assert msg.command == "run"
        assert msg.arg == "NVDA"

    def test_help_command_no_arg(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="/help")
        assert msg.command == "help"
        assert msg.arg == ""

    def test_run_force_word_sets_is_force_and_arg(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="/run NVDA force")
        assert msg.arg == "NVDA"
        assert msg.is_force

    def test_run_double_dash_force_sets_is_force(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="/run nvda --force")
        assert msg.arg == "NVDA"
        assert msg.is_force

    def test_plain_run_is_not_force(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="/run NVDA")
        assert not msg.is_force

    def test_bare_text_not_command(self) -> None:
        msg = _TextMessage(update_id=1, chat_id=1, text="NVDA")
        assert not msg.is_command
        assert msg.command == ""


# ---------------------------------------------------------------------------
# _CallbackTap predicates
# ---------------------------------------------------------------------------


class TestCallbackTapPredicates:
    def test_approve_is_approve(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"approve:{cid}", "alice")
        assert tap.is_approve
        assert not tap.is_reject
        assert tap.correlation_id == cid

    def test_reject_is_reject(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"reject:{cid}", "bob")
        assert tap.is_reject
        assert not tap.is_approve

    def test_act_tap_exposes_verb_and_cid(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"act:buy:{cid}", "carol")
        assert tap.is_action
        assert tap.verb == "buy"
        assert tap.correlation_id == cid

    def test_non_act_tap_has_empty_verb(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"approve:{cid}", "dave")
        assert tap.verb == ""


# ---------------------------------------------------------------------------
# _decision_for — tap → structured HITLDecision
# ---------------------------------------------------------------------------


class TestDecisionFor:
    def test_approve_tap_maps_to_approve(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"approve:{cid}", "alice")
        assert _decision_for(tap, "buy") is HITLDecision.APPROVE

    def test_act_buy_maps_to_override_buy(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"act:buy:{cid}", "alice")
        assert _decision_for(tap, "hold") is HITLDecision.OVERRIDE_BUY

    def test_act_sell_maps_to_override_sell(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"act:sell:{cid}", "alice")
        assert _decision_for(tap, "buy") is HITLDecision.OVERRIDE_SELL

    def test_act_hold_maps_to_override_hold(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"act:hold:{cid}", "alice")
        assert _decision_for(tap, "buy") is HITLDecision.OVERRIDE_HOLD


# ---------------------------------------------------------------------------
# _normalize_recommendation
# ---------------------------------------------------------------------------


class TestNormalizeRecommendation:
    def test_sell_synonyms(self) -> None:
        assert _normalize_recommendation("reduce", "buy") == "sell"
        assert _normalize_recommendation("trim", "buy") == "sell"

    def test_hold_synonyms(self) -> None:
        assert _normalize_recommendation("neutral", "buy") == "hold"

    def test_defaults_to_side_then_buy(self) -> None:
        assert _normalize_recommendation(None, "sell") == "sell"
        assert _normalize_recommendation(None, "") == "buy"


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------


class TestCommandRouting:
    def test_run_command_starts_pipeline_thread(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/run NVDA"))
            mock_start.assert_called_once_with("NVDA", 123, force_buy=False)

    def test_run_command_uppercase_ticker(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/run nvda"))
            mock_start.assert_called_once_with("NVDA", 123, force_buy=False)

    def test_run_force_word_sets_force_buy(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/run NVDA force"))
            mock_start.assert_called_once_with("NVDA", 123, force_buy=True)

    def test_demo_command_sets_force_buy(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/demo NVDA"))
            mock_start.assert_called_once_with("NVDA", 123, force_buy=True)

    def test_demo_command_missing_arg_sends_usage(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "/demo"))
            assert "Usage" in mock_send.call_args[0][1]

    def test_run_command_missing_arg_sends_usage(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "/run"))
            assert "Usage" in mock_send.call_args[0][1]

    def test_help_command_sends_help(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_help") as mock_help:
            bot._handle_text(_TextMessage(1, 123, "/help"))
            mock_help.assert_called_once_with(123)

    def test_start_command_sends_help(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_help") as mock_help:
            bot._handle_text(_TextMessage(1, 123, "/start"))
            mock_help.assert_called_once_with(123)

    def test_help_text_mentions_force_and_demo(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._send_help(123)
            text = mock_send.call_args[0][1]
            assert "force" in text.lower()
            assert "/demo" in text

    def test_portfolio_command_calls_send_portfolio(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_portfolio") as mock_p:
            bot._handle_text(_TextMessage(1, 123, "/portfolio"))
            mock_p.assert_called_once_with(123)

    def test_unknown_command_sends_error(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "/foobar"))
            assert "Unknown command" in mock_send.call_args[0][1]

    def test_bare_ticker_starts_run(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "NVDA"))
            mock_start.assert_called_once_with("NVDA", 123)

    def test_bare_non_alpha_text_sends_hint(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "hello world"))
            assert "/run" in mock_send.call_args[0][1]


# ---------------------------------------------------------------------------
# Dispatch routing — callback vs text
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_callback_routes_to_handle_callback(self) -> None:
        bot = _make_bot()
        with (
            patch.object(bot, "_handle_callback") as mock_cb,
            patch.object(bot, "_handle_text") as mock_text,
        ):
            bot._dispatch(_raw_callback(f"approve:{uuid.uuid4()}"))
            mock_cb.assert_called_once()
            mock_text.assert_not_called()

    def test_text_routes_to_handle_text(self) -> None:
        bot = _make_bot()
        with (
            patch.object(bot, "_handle_callback") as mock_cb,
            patch.object(bot, "_handle_text") as mock_text,
        ):
            bot._dispatch(_raw_message("/run NVDA"))
            mock_text.assert_called_once()
            mock_cb.assert_not_called()


# ---------------------------------------------------------------------------
# force_buy threading into the pipeline run
# ---------------------------------------------------------------------------


class TestForceBuyThreading:
    def _state_passed_to_graph(self, graph: MagicMock) -> dict[str, Any]:
        return graph.stream.call_args[0][0]

    def _idle_graph(self) -> MagicMock:
        graph = MagicMock()
        graph.stream.return_value = iter([{}])
        run_state = MagicMock()
        run_state.next = None
        run_state.tasks = None
        graph.get_state.return_value = run_state
        return graph

    def test_demo_threads_force_buy_true_into_state(self) -> None:
        graph = self._idle_graph()
        bot = _make_bot(graph=graph)
        bot._run_pipeline("NVDA", 123, force_buy=True)
        assert self._state_passed_to_graph(graph)["force_buy"] is True

    def test_plain_run_threads_force_buy_false_into_state(self) -> None:
        graph = self._idle_graph()
        bot = _make_bot(graph=graph)
        bot._run_pipeline("NVDA", 123)
        assert self._state_passed_to_graph(graph)["force_buy"] is False

    def test_start_run_forwards_force_buy_to_thread(self) -> None:
        bot = _make_bot()
        captured: dict[str, Any] = {}

        class _FakeThread:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            def start(self) -> None:
                pass

        with patch("firm.bot.service.threading.Thread", _FakeThread):
            bot._start_run("NVDA", 123, force_buy=True)

        assert captured["kwargs"] == {"force_buy": True}


# ---------------------------------------------------------------------------
# Callback → resume dispatch
# ---------------------------------------------------------------------------


class TestCallbackDispatch:
    def test_approve_wakes_run_with_approve_decision(self) -> None:
        bot = _make_bot()
        cid = str(uuid.uuid4())
        signal = _ResumeSignal()
        bot._pending[cid] = (_pending(cid), signal)

        tap = _CallbackTap(2, "cq-1", 123, f"approve:{cid}", "alice")
        with patch.object(bot, "_answer_callback") as mock_answer:
            bot._handle_callback(tap)

        assert signal.event.is_set()
        assert signal.hitl_status == HITLStatus.APPROVED
        assert signal.resume_value == HITLDecision.APPROVE.value
        mock_answer.assert_called_once()

    def test_reject_does_not_wake_run_and_sends_alternatives(self) -> None:
        bot = _make_bot()
        cid = str(uuid.uuid4())
        signal = _ResumeSignal()
        bot._pending[cid] = (_pending(cid, recommendation="buy"), signal)

        tap = _CallbackTap(2, "cq-1", 123, f"reject:{cid}", "bob")
        with (
            patch.object(bot, "_answer_callback") as mock_answer,
            patch.object(bot, "_send_message_with_keyboard") as mock_kb,
        ):
            bot._handle_callback(tap)

        # Reject is NOT terminal — the run thread keeps waiting.
        assert not signal.event.is_set()
        mock_answer.assert_called_once()
        mock_kb.assert_called_once()
        # The alternatives keyboard offers the OTHER actions (hold/sell for a buy rec).
        keyboard = mock_kb.call_args[0][2]
        verbs = {btn["callback_data"].split(":")[1] for row in keyboard for btn in row}
        assert verbs == {"hold", "sell"}

    def test_act_tap_wakes_run_with_override_decision(self) -> None:
        bot = _make_bot()
        cid = str(uuid.uuid4())
        signal = _ResumeSignal()
        bot._pending[cid] = (_pending(cid, recommendation="buy"), signal)

        tap = _CallbackTap(3, "cq-2", 123, f"act:sell:{cid}", "alice")
        with patch.object(bot, "_answer_callback"):
            bot._handle_callback(tap)

        assert signal.event.is_set()
        assert signal.resume_value == HITLDecision.OVERRIDE_SELL.value
        assert signal.hitl_status == HITLStatus.APPROVED

    def test_unknown_cid_acks_gracefully(self) -> None:
        bot = _make_bot()
        tap = _CallbackTap(2, "cq-1", 123, f"approve:{uuid.uuid4()}", "eve")
        with patch.object(bot, "_answer_callback") as mock_answer:
            bot._handle_callback(tap)
        mock_answer.assert_called_once()
        text = mock_answer.call_args[0][1]
        assert "expired" in text.lower() or "No matching" in text


# ---------------------------------------------------------------------------
# Approval card formatter
# ---------------------------------------------------------------------------


class TestApprovalCardFormatter:
    def _make_req_dict(self, **kwargs: Any) -> dict[str, Any]:
        cid = str(uuid.uuid4())
        base = {
            "trade_id": str(uuid.uuid4()),
            "symbol": "NVDA",
            "side": "buy",
            "qty_str": "49",
            "notional": "9883.30",
            "reason": "Trade notional 9.88% exceeds HITL threshold 5.00%",
            "expires_at": (datetime.now(tz=UTC) + timedelta(minutes=10)).isoformat(),
            "correlation_id": cid,
            "recommendation": "buy",
            "conviction": 0.85,
            "rationale": "Strong earnings momentum and beat estimates.",
            "bull_case": "AI chip demand remains structurally elevated.",
            "bear_case": "Valuation stretched; macro headwinds possible.",
        }
        base.update(kwargs)
        return base

    def test_card_contains_symbol_and_side(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "NVDA" in card
        assert "BUY" in card

    def test_card_contains_recommendation(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "Buy" in card  # title-cased

    def test_card_contains_conviction(self) -> None:
        assert "85%" in format_approval_card(self._make_req_dict())

    def test_card_contains_rationale(self) -> None:
        assert "Strong earnings momentum" in format_approval_card(self._make_req_dict())

    def test_card_contains_bull_and_bear(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "AI chip demand" in card
        assert "Valuation stretched" in card

    def test_card_contains_notional(self) -> None:
        assert "9,883.30" in format_approval_card(self._make_req_dict())

    def test_real_buy_renders_trade_line(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "BUY 49 NVDA" in card

    def test_hold_proposal_has_no_phantom_buy(self) -> None:
        # qty=0, notional=0, recommendation=hold → "no trade to place", never "BUY 0".
        card = format_approval_card(
            self._make_req_dict(qty_str="0", notional="0", side="hold", recommendation="hold")
        )
        assert "no trade to place" in card
        assert "BUY 0" not in card

    def test_card_without_research_context_still_renders(self) -> None:
        card = format_approval_card(
            self._make_req_dict(
                recommendation=None,
                conviction=None,
                rationale=None,
                bull_case=None,
                bear_case=None,
            )
        )
        assert "NVDA" in card
        assert "Approve" in card

    def test_card_contains_approve_hint(self) -> None:
        assert "Approve" in format_approval_card(self._make_req_dict())


# ---------------------------------------------------------------------------
# Decision report formatter
# ---------------------------------------------------------------------------


class TestDecisionReportFormatter:
    def _fill_state(self, side: str = "buy") -> dict[str, Any]:
        return {
            "cycle_outcome": "filled",
            "synthesis": {
                "executive_summary": "NVDA showed strong momentum driven by AI demand.",
                "title": "NVDA Earnings Day Report",
            },
            "verdict": {"coherence_score": 4, "alignment": "aligned"},
            "approved_trade": {
                "trade": {"qty": "49", "requested_price": "201.70", "side": side},
            },
        }

    def test_approve_fill_shows_approved_filled(self) -> None:
        state = self._fill_state()
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome=state["cycle_outcome"],
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
            hitl_decision=HITLDecision.APPROVE.value,
        )
        assert "✅ *Approved → FILLED" in report
        assert "NVDA" in report
        assert "201.70" in report
        assert "AI demand" in report
        assert "4/5" in report

    def test_override_buy_shows_bought(self) -> None:
        state = self._fill_state(side="buy")
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome="filled",
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
            hitl_decision=HITLDecision.OVERRIDE_BUY.value,
            recommendation="hold",
        )
        assert "🟢 *Override → Bought" in report
        assert "You overrode the desk's hold" in report

    def test_override_sell_shows_sold(self) -> None:
        state = self._fill_state(side="sell")
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome="filled",
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
            hitl_decision=HITLDecision.OVERRIDE_SELL.value,
            recommendation="buy",
        )
        assert "🔴 *Override → Sold" in report
        assert "You overrode the desk's buy" in report

    def test_override_hold_shows_held(self) -> None:
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome="hold",
            synthesis=None,
            verdict=None,
            approved_trade=None,
            hitl_decision=HITLDecision.OVERRIDE_HOLD.value,
            recommendation="buy",
        )
        assert "⏸ *Held (your override)" in report
        assert "You overrode the desk's buy" in report

    def test_rejection_report_contains_rejected_marker(self) -> None:
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="rejected",
            cycle_outcome="rejected",
            synthesis=None,
            verdict=None,
            approved_trade=None,
            rejection_reason="operator rejected",
        )
        assert "Rejected" in report or "❌" in report
        assert "no trade booked" in report

    def test_approved_not_filled_report(self) -> None:
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome="rejected_market_closed",
            synthesis=None,
            verdict=None,
            approved_trade=None,
            hitl_decision=HITLDecision.APPROVE.value,
        )
        assert "Approved" in report or "approved" in report.lower()
        assert "market_closed" in report or "not filled" in report.lower()


# ---------------------------------------------------------------------------
# Portfolio report formatter
# ---------------------------------------------------------------------------


class TestPortfolioReportFormatter:
    def test_shows_nav_and_cash(self) -> None:
        report = format_portfolio_report(
            nav="105,000.00",
            cash="80,000.00",
            pnl="+25,000.00",
            holdings=[{"symbol": "NVDA", "qty": "49"}],
        )
        assert "105,000.00" in report
        assert "80,000.00" in report

    def test_shows_holdings(self) -> None:
        report = format_portfolio_report(
            nav="105,000.00",
            cash="80,000.00",
            pnl="+25,000.00",
            holdings=[{"symbol": "NVDA", "qty": "49"}, {"symbol": "MSFT", "qty": "12"}],
        )
        assert "NVDA" in report
        assert "MSFT" in report

    def test_empty_holdings_shows_no_positions(self) -> None:
        report = format_portfolio_report(
            nav="100,000.00",
            cash="100,000.00",
            pnl="+0.00",
            holdings=[],
        )
        assert "No open positions" in report


# ---------------------------------------------------------------------------
# _build_hitl_request
# ---------------------------------------------------------------------------


class TestBuildHitlRequest:
    def test_builds_from_full_payload(self) -> None:
        cid = str(uuid.uuid4())
        payload = {
            "trade_proposal": {
                "id": str(uuid.uuid4()),
                "symbol": "NVDA",
                "side": "buy",
                "qty": "49",
                "notional": "9883.30",
                "rationale": "Large trade requires review",
            },
            "research_plan": {
                "recommendation": "buy",
                "conviction": 0.85,
                "rationale": "Strong AI demand.",
                "bull_summary": "Bull case text.",
                "bear_summary": "Bear case text.",
            },
        }
        req = _build_hitl_request(payload, cid)
        assert req.symbol == "NVDA"
        assert req.side == "buy"
        assert req.recommendation == "buy"
        assert req.conviction == pytest.approx(0.85)
        assert req.bull_case == "Bull case text."
        assert req.bear_case == "Bear case text."
        assert req.correlation_id == cid

    def test_builds_from_minimal_payload(self) -> None:
        cid = str(uuid.uuid4())
        req = _build_hitl_request({}, cid)
        assert req.symbol == "?"
        assert req.recommendation is None
        assert req.conviction is None


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------


class TestApprovalKeyboard:
    def test_keyboard_has_approve_and_reject(self) -> None:
        cid = str(uuid.uuid4())
        all_data = [btn["callback_data"] for row in _approval_keyboard(cid) for btn in row]
        assert any(d.startswith("approve:") for d in all_data)
        assert any(d.startswith("reject:") for d in all_data)

    def test_keyboard_encodes_correlation_id(self) -> None:
        cid = str(uuid.uuid4())
        all_data = [btn["callback_data"] for row in _approval_keyboard(cid) for btn in row]
        assert f"approve:{cid}" in all_data
        assert f"reject:{cid}" in all_data


class TestAlternativesKeyboard:
    def test_buy_rec_offers_hold_and_sell(self) -> None:
        cid = str(uuid.uuid4())
        keyboard = _alternatives_keyboard("buy", cid)
        data = [btn["callback_data"] for row in keyboard for btn in row]
        assert f"act:hold:{cid}" in data
        assert f"act:sell:{cid}" in data

    def test_sell_rec_offers_buy_and_hold(self) -> None:
        cid = str(uuid.uuid4())
        keyboard = _alternatives_keyboard("sell", cid)
        verbs = {btn["callback_data"].split(":")[1] for row in keyboard for btn in row}
        assert verbs == {"buy", "hold"}


# ---------------------------------------------------------------------------
# CLI — 'bot' subcommand is registered
# ---------------------------------------------------------------------------


class TestCliBotSubcommand:
    def test_bot_subcommand_exists(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args, _ = parser.parse_known_args(["bot"])
        assert args.command == "bot"
