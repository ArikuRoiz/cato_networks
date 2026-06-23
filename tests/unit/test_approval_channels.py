"""Tests for the pluggable approval-channel registry and adapters.

Covers:
- ``build_approval_channel`` resolves each registered name and the ``auto``
  sentinel (telegram when configured, else console).
- Unknown names raise a clear ``ValueError``.
- Constructed channels satisfy the ``ApprovalChannel`` Protocol.
- ``TelegramApprovalChannel`` maps approval-result statuses to decisions.
- ``SlackApprovalChannel`` sends the card and fails safe to EXPIRE (receive side
  not yet wired).

No network: Telegram/Slack adapters run in dry-run mode (no credentials) or are
exercised through mocks.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from firm.adapters.approval import (
    AVAILABLE_CHANNELS,
    ConsoleApprovalChannel,
    SlackApprovalChannel,
    TelegramApprovalChannel,
    build_approval_channel,
)
from firm.config.settings import Settings
from firm.domain.enums import HITLStatus
from firm.orchestration.hitl import HITLDecision
from firm.ports.approval import ApprovalChannel
from firm.ports.types import ApprovalResult, HITLRequest


def _settings(*, telegram: bool = False) -> Settings:
    token = "123:realtoken" if telegram else ""
    chat_id = "555" if telegram else ""
    return Settings(
        database_url="postgresql://x",
        anthropic_api_key="",
        slack_bot_token="",
        slack_channel="#trading-desk",
        telegram_bot_token=token,
        telegram_chat_id=chat_id,
        langfuse_public_key="",
        langfuse_secret_key="",
        otel_endpoint="",
    )


def _request() -> HITLRequest:
    return HITLRequest(
        trade_id=uuid.uuid4(),
        symbol="NVDA",
        side="buy",
        qty_str="10",
        notional=Decimal("1000"),
        reason="test",
        expires_at=datetime.now(tz=UTC) + timedelta(minutes=10),
        correlation_id="cid-1",
    )


# ---------------------------------------------------------------------------
# Registry / selection
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_available_channels_lists_choices(self) -> None:
        assert set(AVAILABLE_CHANNELS) == {"console", "telegram", "slack", "auto"}

    def test_console_name_resolves_to_console_channel(self) -> None:
        channel = build_approval_channel("console", _settings())
        assert isinstance(channel, ConsoleApprovalChannel)

    def test_telegram_name_resolves_to_telegram_channel(self) -> None:
        channel = build_approval_channel("telegram", _settings(telegram=True))
        assert isinstance(channel, TelegramApprovalChannel)

    def test_slack_name_resolves_to_slack_channel(self) -> None:
        channel = build_approval_channel("slack", _settings())
        assert isinstance(channel, SlackApprovalChannel)

    def test_auto_picks_telegram_when_configured(self) -> None:
        channel = build_approval_channel("auto", _settings(telegram=True))
        assert isinstance(channel, TelegramApprovalChannel)

    def test_auto_falls_back_to_console_without_telegram(self) -> None:
        channel = build_approval_channel("auto", _settings(telegram=False))
        assert isinstance(channel, ConsoleApprovalChannel)

    def test_unknown_name_raises_value_error_listing_choices(self) -> None:
        with pytest.raises(ValueError, match="Unknown HITL channel 'sms'"):
            build_approval_channel("sms", _settings())

    def test_every_registered_channel_satisfies_protocol(self) -> None:
        for name in ("console", "telegram", "slack"):
            channel = build_approval_channel(name, _settings(telegram=True))
            assert isinstance(channel, ApprovalChannel)


# ---------------------------------------------------------------------------
# Telegram channel — status → decision mapping
# ---------------------------------------------------------------------------


class TestTelegramApprovalChannel:
    def _channel_with_status(self, status: HITLStatus) -> TelegramApprovalChannel:
        hitl = MagicMock()
        hitl.send_hitl_request.return_value = ApprovalResult(status=status)
        return TelegramApprovalChannel(hitl)

    def test_approved_maps_to_approve(self) -> None:
        channel = self._channel_with_status(HITLStatus.APPROVED)
        assert channel.request_decision(_request()) is HITLDecision.APPROVE

    def test_rejected_maps_to_override_hold(self) -> None:
        channel = self._channel_with_status(HITLStatus.REJECTED)
        assert channel.request_decision(_request()) is HITLDecision.OVERRIDE_HOLD

    def test_expired_maps_to_expire(self) -> None:
        channel = self._channel_with_status(HITLStatus.EXPIRED)
        assert channel.request_decision(_request()) is HITLDecision.EXPIRE


# ---------------------------------------------------------------------------
# Slack channel — send side wired, receive side stubbed
# ---------------------------------------------------------------------------


class TestSlackApprovalChannel:
    def test_sends_card_and_fails_safe_to_expire(self) -> None:
        sink = MagicMock()
        channel = SlackApprovalChannel(sink)
        req = _request()
        decision = channel.request_decision(req)
        sink.send_hitl_request.assert_called_once_with(req)
        assert decision is HITLDecision.EXPIRE

    def test_raise_on_receive_surfaces_integration_gap(self) -> None:
        channel = SlackApprovalChannel(MagicMock(), raise_on_receive=True)
        with pytest.raises(NotImplementedError, match="webhook"):
            channel.request_decision(_request())
