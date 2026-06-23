"""Approval-channel registry: ``--hitl <name>`` → constructed ``ApprovalChannel``.

Adding a channel is: write an adapter that implements ``ApprovalChannel``, then
register a factory here. ``auto`` resolves to telegram when configured, else
console.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from firm.adapters.approval.console import ConsoleApprovalChannel
from firm.adapters.approval.slack import SlackApprovalChannel
from firm.adapters.approval.telegram import TelegramApprovalChannel

if TYPE_CHECKING:
    from firm.config.settings import Settings
    from firm.ports.approval import ApprovalChannel

# name → factory(settings) -> ApprovalChannel
ChannelFactory = Callable[["Settings"], "ApprovalChannel"]

_REGISTRY: dict[str, ChannelFactory] = {
    "console": lambda _settings: ConsoleApprovalChannel(),
    "telegram": lambda settings: TelegramApprovalChannel.from_credentials(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    ),
    "slack": lambda settings: SlackApprovalChannel.from_channel(settings.slack_channel),
}

# Selectable names for ``--hitl`` (registered channels + the auto sentinel).
AVAILABLE_CHANNELS: tuple[str, ...] = (*_REGISTRY.keys(), "auto")


def build_approval_channel(name: str, settings: Settings) -> ApprovalChannel:
    """Resolve a ``--hitl`` *name* to a constructed approval channel.

    ``auto`` → telegram when ``settings.has_telegram`` else console. Unknown
    names raise ``ValueError`` listing the valid choices.
    """
    resolved = _resolve_auto(settings) if name == "auto" else name
    factory = _REGISTRY.get(resolved)
    if factory is None:
        raise ValueError(
            f"Unknown HITL channel '{name}'. Choices: {', '.join(AVAILABLE_CHANNELS)}."
        )
    return factory(settings)


def _resolve_auto(settings: Settings) -> str:
    """Pick the best default channel: telegram when configured, else console."""
    return "telegram" if settings.has_telegram else "console"
