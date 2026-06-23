"""Unit tests for TelegramHITL adapter.

All tests use mocked httpx.Client — no live Telegram calls are made.

Coverage:
- sendMessage payload shape + inline keyboard layout.
- getUpdates long-poll parses a callback_query into APPROVED / REJECTED.
- Timeout / expiry → EXPIRED (fail-safe, never auto-approve).
- Disabled / dry-run mode (missing or placeholder credentials) → EXPIRED.
- _parse_callback_update extracts fields correctly.
- _matches_request filters by correlation_id.
- _map_callback_to_result maps approve / reject / edit / unknown actions.
- CLI --hitl and --force-buy arg parsing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from firm.adapters.telegram import (
    TelegramHITL,
    _CallbackUpdate,
    _inline_keyboard,
    _map_callback_to_result,
    _matches_request,
    _parse_callback_update,
)
from firm.domain.enums import ApprovalStatus
from firm.ports.types import HITLRequest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_req(
    correlation_id: str | None = None,
    expires_in_seconds: float = 300,
) -> HITLRequest:
    cid = correlation_id or str(uuid.uuid4())
    return HITLRequest(
        trade_id=uuid.uuid4(),
        symbol="NVDA",
        side="buy",
        qty_str="10",
        notional=Decimal("15000.00"),
        reason="Large BUY exceeds 5% NAV threshold",
        expires_at=datetime.now(tz=UTC) + timedelta(seconds=expires_in_seconds),
        correlation_id=cid,
    )


def _callback_update(action: str, correlation_id: str, update_id: int = 1) -> dict:  # type: ignore[type-arg]
    """Build a raw Telegram update dict containing a callback_query."""
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cq-123",
            "data": f"{action}:{correlation_id}",
            "from": {"id": 42, "username": "risk_officer"},
        },
    }


def _make_http_client(responses: list[dict]) -> MagicMock:  # type: ignore[type-arg]
    """Build a mock httpx.Client whose post() returns responses in sequence."""
    client = MagicMock()
    mock_responses = []
    for body in responses:
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status = MagicMock()
        mock_responses.append(resp)
    client.post.side_effect = mock_responses
    return client


# ---------------------------------------------------------------------------
# Dry-run / disabled mode
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_no_token_returns_expired(self) -> None:
        hitl = TelegramHITL(token="", chat_id="123")
        result = hitl.send_hitl_request(_make_req())
        assert result.status == ApprovalStatus.EXPIRED

    def test_placeholder_token_returns_expired(self) -> None:
        hitl = TelegramHITL(token="123456:ABC-...", chat_id="123")
        result = hitl.send_hitl_request(_make_req())
        assert result.status == ApprovalStatus.EXPIRED

    def test_no_chat_id_returns_expired(self) -> None:
        hitl = TelegramHITL(token="123456:ABC-DEF", chat_id="")
        result = hitl.send_hitl_request(_make_req())
        assert result.status == ApprovalStatus.EXPIRED

    def test_dry_run_never_calls_http(self) -> None:
        http = MagicMock()
        hitl = TelegramHITL(token="", chat_id="", http_client=http)
        hitl.send_hitl_request(_make_req())
        http.post.assert_not_called()

    def test_already_expired_request_returns_expired(self) -> None:
        http = MagicMock()
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        expired_req = _make_req(expires_in_seconds=-60)  # already past
        result = hitl.send_hitl_request(expired_req)
        assert result.status == ApprovalStatus.EXPIRED
        http.post.assert_not_called()


# ---------------------------------------------------------------------------
# sendMessage payload shape
# ---------------------------------------------------------------------------


class TestSendMessagePayload:
    def test_sendmessage_contains_symbol_and_side(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)

        # sendMessage → ok; getUpdates immediately returns matching approve
        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},  # sendMessage
                {"ok": True, "result": [_callback_update("approve", cid)]},  # getUpdates
                {"ok": True, "result": {}},  # answerCallbackQuery
                {"ok": True, "result": {"message_id": 2}},  # confirmation sendMessage
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        hitl.send_hitl_request(req)

        first_call = http.post.call_args_list[0]
        payload = first_call[1]["json"]  # keyword arg 'json'
        assert "NVDA" in payload["text"]
        assert "BUY" in payload["text"]

    def test_sendmessage_includes_inline_keyboard(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                {"ok": True, "result": [_callback_update("reject", cid)]},
                {"ok": True, "result": {}},
                {"ok": True, "result": {"message_id": 2}},
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        hitl.send_hitl_request(req)

        payload = http.post.call_args_list[0][1]["json"]
        keyboard = payload["reply_markup"]["inline_keyboard"]
        # First row: Approve + Reject
        assert len(keyboard[0]) == 2
        approve_btn = keyboard[0][0]
        reject_btn = keyboard[0][1]
        assert f"approve:{cid}" == approve_btn["callback_data"]
        assert f"reject:{cid}" == reject_btn["callback_data"]

    def test_inline_keyboard_edit_button_present(self) -> None:
        req = _make_req()
        keyboard = _inline_keyboard(req)
        all_data = [btn["callback_data"] for row in keyboard for btn in row]
        assert any(d.startswith("edit:") for d in all_data)


# ---------------------------------------------------------------------------
# getUpdates polling → APPROVED / REJECTED
# ---------------------------------------------------------------------------


class TestPollApproved:
    def test_approve_callback_returns_approved(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                {"ok": True, "result": [_callback_update("approve", cid)]},
                {"ok": True, "result": {}},
                {"ok": True, "result": {"message_id": 2}},
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        result = hitl.send_hitl_request(req)

        assert result.status == ApprovalStatus.APPROVED
        assert result.decided_by == "risk_officer"

    def test_reject_callback_returns_rejected(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                {"ok": True, "result": [_callback_update("reject", cid)]},
                {"ok": True, "result": {}},
                {"ok": True, "result": {"message_id": 2}},
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        result = hitl.send_hitl_request(req)

        assert result.status == ApprovalStatus.REJECTED

    def test_edit_callback_treated_as_approved(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                {"ok": True, "result": [_callback_update("edit", cid)]},
                {"ok": True, "result": {}},
                {"ok": True, "result": {"message_id": 2}},
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)
        result = hitl.send_hitl_request(req)

        assert result.status == ApprovalStatus.APPROVED

    def test_ignores_callback_for_different_correlation_id(self) -> None:
        """Callbacks for other requests are ignored; expiry returns EXPIRED."""
        cid = str(uuid.uuid4())
        other_cid = str(uuid.uuid4())
        # Use a near-future expiry so the poll exits quickly
        req = _make_req(correlation_id=cid, expires_in_seconds=1)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                # Callback for a different correlation_id — must be ignored
                {"ok": True, "result": [_callback_update("approve", other_cid)]},
                # Second poll; empty result → loop exits when expires_at passes
                {"ok": True, "result": []},
                {"ok": True, "result": {"message_id": 2}},  # expiry message
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)

        with patch("firm.adapters.telegram.time.sleep"):
            result = hitl.send_hitl_request(req)

        assert result.status == ApprovalStatus.EXPIRED


# ---------------------------------------------------------------------------
# Timeout → EXPIRED (fail-safe)
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_expired_request_returns_expired_not_approved(self) -> None:
        """An expired poll must return EXPIRED, never APPROVED."""
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid, expires_in_seconds=1)

        http = _make_http_client(
            [
                {"ok": True, "result": {"message_id": 1}},
                {"ok": True, "result": []},  # empty updates → expiry loop
                {"ok": True, "result": {"message_id": 2}},  # expiry notice
            ]
        )
        hitl = TelegramHITL(token="123:REAL", chat_id="-100123", http_client=http)

        with patch("firm.adapters.telegram.time.sleep"):
            result = hitl.send_hitl_request(req)

        assert result.status == ApprovalStatus.EXPIRED
        assert result.status != ApprovalStatus.APPROVED  # explicit fail-safe assertion


# ---------------------------------------------------------------------------
# _parse_callback_update
# ---------------------------------------------------------------------------


class TestParseCallbackUpdate:
    def test_parses_valid_callback_update(self) -> None:
        raw = _callback_update("approve", "cid-abc")
        parsed = _parse_callback_update(raw)
        assert parsed is not None
        assert parsed.callback_data == "approve:cid-abc"
        assert parsed.from_username == "risk_officer"
        assert parsed.update_id == 1

    def test_returns_none_for_non_callback_update(self) -> None:
        raw = {"update_id": 1, "message": {"text": "hello"}}
        assert _parse_callback_update(raw) is None

    def test_returns_none_when_callback_query_missing_data(self) -> None:
        raw = {
            "update_id": 1,
            "callback_query": {"id": "cq-1", "from": {"username": "u"}},
            # "data" missing
        }
        assert _parse_callback_update(raw) is None


# ---------------------------------------------------------------------------
# _matches_request
# ---------------------------------------------------------------------------


class TestMatchesRequest:
    def test_matching_correlation_id_returns_true(self) -> None:
        cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)
        cb = _CallbackUpdate(1, "cq1", f"approve:{cid}", "user")
        assert _matches_request(cb, req) is True

    def test_different_correlation_id_returns_false(self) -> None:
        cid = str(uuid.uuid4())
        other_cid = str(uuid.uuid4())
        req = _make_req(correlation_id=cid)
        cb = _CallbackUpdate(1, "cq1", f"approve:{other_cid}", "user")
        assert _matches_request(cb, req) is False

    def test_malformed_callback_data_returns_false(self) -> None:
        req = _make_req()
        cb = _CallbackUpdate(1, "cq1", "no-colon-here", "user")
        assert _matches_request(cb, req) is False


# ---------------------------------------------------------------------------
# _map_callback_to_result
# ---------------------------------------------------------------------------


class TestMapCallbackToResult:
    def test_approve_maps_to_approved(self) -> None:
        cid = str(uuid.uuid4())
        cb = _CallbackUpdate(1, "cq1", f"approve:{cid}", "alice")
        result = _map_callback_to_result(cb)
        assert result.status == ApprovalStatus.APPROVED
        assert result.decided_by == "alice"

    def test_reject_maps_to_rejected(self) -> None:
        cid = str(uuid.uuid4())
        cb = _CallbackUpdate(1, "cq1", f"reject:{cid}", "bob")
        result = _map_callback_to_result(cb)
        assert result.status == ApprovalStatus.REJECTED

    def test_edit_maps_to_approved(self) -> None:
        cid = str(uuid.uuid4())
        cb = _CallbackUpdate(1, "cq1", f"edit:{cid}", "carol")
        result = _map_callback_to_result(cb)
        assert result.status == ApprovalStatus.APPROVED

    def test_unknown_action_maps_to_rejected(self) -> None:
        cid = str(uuid.uuid4())
        cb = _CallbackUpdate(1, "cq1", f"unknown:{cid}", "dave")
        result = _map_callback_to_result(cb)
        assert result.status == ApprovalStatus.REJECTED

    def test_no_username_uses_fallback(self) -> None:
        cid = str(uuid.uuid4())
        cb = _CallbackUpdate(1, "cq1", f"approve:{cid}", None)
        result = _map_callback_to_result(cb)
        assert result.decided_by == "telegram-user"


# ---------------------------------------------------------------------------
# CLI arg parsing — --hitl and --force-buy
# ---------------------------------------------------------------------------


class TestCliHitlArgs:
    def test_run_hitl_default_is_auto(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.hitl == "auto"

    def test_run_hitl_console(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--hitl", "console"])
        assert args.hitl == "console"

    def test_run_hitl_telegram(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--hitl", "telegram"])
        assert args.hitl == "telegram"

    def test_run_hitl_rejects_invalid_choice(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--hitl", "fax"])

    def test_run_force_buy_default_false(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.force_buy is False

    def test_run_force_buy_flag_sets_true(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--force-buy"])
        assert args.force_buy is True

    def test_run_force_buy_combined_with_tickers(self) -> None:
        from firm.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--force-buy", "--tickers", "NVDA", "--hitl", "telegram"])
        assert args.force_buy is True
        assert args.tickers == "NVDA"
        assert args.hitl == "telegram"
