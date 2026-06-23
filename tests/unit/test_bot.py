"""Unit tests for the Telegram bot service.

All tests use mocked httpx.Client and a mocked graph — no live network,
no live Postgres, no live LangGraph execution.

Coverage
--------
- Command routing: /run TICKER, bare ticker, /help, /start, /portfolio, unknown
- Callback → resume dispatch (approve, reject, unknown CID)
- Rich-card formatter (research_plan → human pros/cons)
- Post-decision explanation builder (fill, approved-not-filled, rejected)
- Portfolio report builder
- parse_text_message / parse_callback_tap boundary wrappers
- _map_tap_to_decision
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
    _TextMessage,
    parse_callback_tap,
    parse_text_message,
)
from firm.bot.service import (
    BotService,
    _approval_keyboard,
    _build_hitl_request,
    _map_tap_to_decision,
)
from firm.domain.enums import HITLStatus

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


def _ok_response(body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"ok": True, "result": body or {}}
    resp.raise_for_status = MagicMock()
    return resp


def _raw_message(text: str, chat_id: int = 123) -> dict[str, Any]:
    return {
        "update_id": 1,
        "message": {"text": text, "chat": {"id": chat_id}},
    }


def _raw_callback(action: str, cid: str, chat_id: int = 123, update_id: int = 2) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cq-1",
            "data": f"{action}:{cid}",
            "from": {"username": "risk_officer"},
            "message": {"chat": {"id": chat_id}},
        },
    }


# ---------------------------------------------------------------------------
# parse_text_message
# ---------------------------------------------------------------------------


class TestParseTextMessage:
    def test_parses_command(self) -> None:
        raw = _raw_message("/run NVDA")
        msg = parse_text_message(raw)
        assert msg is not None
        assert msg.is_command
        assert msg.command == "run"
        assert msg.arg == "NVDA"

    def test_parses_bare_text(self) -> None:
        raw = _raw_message("hello world")
        msg = parse_text_message(raw)
        assert msg is not None
        assert not msg.is_command

    def test_returns_none_for_callback(self) -> None:
        raw = _raw_callback("approve", "cid-123")
        assert parse_text_message(raw) is None

    def test_returns_none_for_missing_text(self) -> None:
        raw = {"update_id": 1, "message": {"chat": {"id": 1}}}
        assert parse_text_message(raw) is None


# ---------------------------------------------------------------------------
# parse_callback_tap
# ---------------------------------------------------------------------------


class TestParseCallbackTap:
    def test_parses_approve(self) -> None:
        cid = str(uuid.uuid4())
        raw = _raw_callback("approve", cid)
        tap = parse_callback_tap(raw)
        assert tap is not None
        assert tap.is_approve
        assert tap.correlation_id == cid
        assert tap.from_username == "risk_officer"

    def test_parses_reject(self) -> None:
        cid = str(uuid.uuid4())
        raw = _raw_callback("reject", cid)
        tap = parse_callback_tap(raw)
        assert tap is not None
        assert tap.is_reject

    def test_returns_none_for_text_message(self) -> None:
        raw = _raw_message("/run NVDA")
        assert parse_callback_tap(raw) is None

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


# ---------------------------------------------------------------------------
# _map_tap_to_decision
# ---------------------------------------------------------------------------


class TestMapTapToDecision:
    def test_approve_tap_returns_approved(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"approve:{cid}", "alice")
        status, resume = _map_tap_to_decision(tap)
        assert status == HITLStatus.APPROVED
        assert resume == "approved"

    def test_reject_tap_returns_rejected(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"reject:{cid}", "bob")
        status, resume = _map_tap_to_decision(tap)
        assert status == HITLStatus.REJECTED
        assert resume == "rejected"

    def test_unknown_action_treated_as_rejected(self) -> None:
        cid = str(uuid.uuid4())
        tap = _CallbackTap(1, "cq1", 1, f"unknown:{cid}", "charlie")
        status, _resume = _map_tap_to_decision(tap)
        assert status == HITLStatus.REJECTED


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------


class TestCommandRouting:
    def test_run_command_starts_pipeline_thread(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/run NVDA"))
            mock_start.assert_called_once_with("NVDA", 123)

    def test_run_command_uppercase_ticker(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "/run nvda"))
            mock_start.assert_called_once_with("NVDA", 123)

    def test_run_command_missing_arg_sends_usage(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "/run"))
            text = mock_send.call_args[0][1]
            assert "Usage" in text

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

    def test_portfolio_command_calls_send_portfolio(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_portfolio") as mock_p:
            bot._handle_text(_TextMessage(1, 123, "/portfolio"))
            mock_p.assert_called_once_with(123)

    def test_unknown_command_sends_error(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "/foobar"))
            text = mock_send.call_args[0][1]
            assert "Unknown command" in text

    def test_bare_ticker_starts_run(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_start_run") as mock_start:
            bot._handle_text(_TextMessage(1, 123, "NVDA"))
            mock_start.assert_called_once_with("NVDA", 123)

    def test_bare_non_alpha_text_sends_hint(self) -> None:
        bot = _make_bot()
        with patch.object(bot, "_send_text") as mock_send:
            bot._handle_text(_TextMessage(1, 123, "hello world"))
            text = mock_send.call_args[0][1]
            assert "/run" in text


# ---------------------------------------------------------------------------
# Callback → resume dispatch
# ---------------------------------------------------------------------------


class TestCallbackDispatch:
    def test_approve_resumes_matching_run(self) -> None:
        bot = _make_bot()
        cid = str(uuid.uuid4())
        from firm.bot.schemas import _PendingRun
        from firm.bot.service import _ResumeSignal

        pending = _PendingRun(cid, "tid-1", 123, "NVDA")
        signal = _ResumeSignal()
        bot._pending[cid] = (pending, signal)

        tap = _CallbackTap(2, "cq-1", 123, f"approve:{cid}", "alice")
        with patch.object(bot, "_answer_callback") as mock_answer:
            bot._handle_callback(tap)

        assert signal.event.is_set()
        assert signal.hitl_status == HITLStatus.APPROVED
        assert signal.resume_value == "approved"
        mock_answer.assert_called_once()

    def test_reject_sets_rejected_status(self) -> None:
        bot = _make_bot()
        cid = str(uuid.uuid4())
        from firm.bot.schemas import _PendingRun
        from firm.bot.service import _ResumeSignal

        pending = _PendingRun(cid, "tid-1", 123, "NVDA")
        signal = _ResumeSignal()
        bot._pending[cid] = (pending, signal)

        tap = _CallbackTap(2, "cq-1", 123, f"reject:{cid}", "bob")
        bot._handle_callback(tap)

        assert signal.hitl_status == HITLStatus.REJECTED
        assert signal.resume_value == "rejected"

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
        card = format_approval_card(self._make_req_dict())
        assert "85%" in card

    def test_card_contains_rationale(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "Strong earnings momentum" in card

    def test_card_contains_bull_and_bear(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "AI chip demand" in card
        assert "Valuation stretched" in card

    def test_card_contains_notional(self) -> None:
        card = format_approval_card(self._make_req_dict())
        assert "9,883.30" in card

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
        card = format_approval_card(self._make_req_dict())
        assert "Approve" in card


# ---------------------------------------------------------------------------
# Decision report formatter
# ---------------------------------------------------------------------------


class TestDecisionReportFormatter:
    def _make_fill_state(self) -> dict[str, Any]:
        return {
            "cycle_outcome": "filled",
            "synthesis": {
                "executive_summary": "NVDA showed strong momentum driven by AI demand.",
                "title": "NVDA Earnings Day Report",
            },
            "verdict": {"coherence_score": 4, "alignment": "aligned"},
            "approved_trade": {
                "trade": {"qty": "49", "requested_price": "201.70"},
            },
        }

    def test_fill_report_contains_filled_marker(self) -> None:
        state = self._make_fill_state()
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome=state["cycle_outcome"],
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
        )
        assert "FILLED" in report
        assert "NVDA" in report

    def test_fill_report_contains_price(self) -> None:
        state = self._make_fill_state()
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome=state["cycle_outcome"],
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
        )
        assert "201.70" in report

    def test_fill_report_contains_executive_summary(self) -> None:
        state = self._make_fill_state()
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome=state["cycle_outcome"],
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
        )
        assert "AI demand" in report

    def test_fill_report_contains_judge_score(self) -> None:
        state = self._make_fill_state()
        report = format_decision_report(
            symbol="NVDA",
            hitl_status="approved",
            cycle_outcome=state["cycle_outcome"],
            synthesis=state["synthesis"],
            verdict=state["verdict"],
            approved_trade=state["approved_trade"],
        )
        assert "4/5" in report

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
# _approval_keyboard
# ---------------------------------------------------------------------------


class TestApprovalKeyboard:
    def test_keyboard_has_approve_and_reject(self) -> None:
        cid = str(uuid.uuid4())
        keyboard = _approval_keyboard(cid)
        all_data = [btn["callback_data"] for row in keyboard for btn in row]
        assert any(d.startswith("approve:") for d in all_data)
        assert any(d.startswith("reject:") for d in all_data)

    def test_keyboard_encodes_correlation_id(self) -> None:
        cid = str(uuid.uuid4())
        keyboard = _approval_keyboard(cid)
        all_data = [btn["callback_data"] for row in keyboard for btn in row]
        assert f"approve:{cid}" in all_data
        assert f"reject:{cid}" in all_data


# ---------------------------------------------------------------------------
# CLI — 'bot' subcommand is registered
# ---------------------------------------------------------------------------


class TestCliBotSubcommand:
    def test_bot_subcommand_exists(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        # parse_known_args lets us confirm 'bot' is valid without needing all envs
        args, _ = parser.parse_known_args(["bot"])
        assert args.command == "bot"
