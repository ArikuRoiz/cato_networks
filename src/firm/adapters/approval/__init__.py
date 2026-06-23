"""Approval-channel adapters and the channel registry.

Each adapter implements :class:`firm.ports.approval.ApprovalChannel`.
:func:`build_approval_channel` resolves a ``--hitl <name>`` selection to a
constructed channel; :data:`AVAILABLE_CHANNELS` names the choices for ``--help``.
"""

from firm.adapters.approval.console import ConsoleApprovalChannel
from firm.adapters.approval.registry import (
    AVAILABLE_CHANNELS,
    build_approval_channel,
)
from firm.adapters.approval.slack import SlackApprovalChannel
from firm.adapters.approval.telegram import TelegramApprovalChannel

__all__ = [
    "AVAILABLE_CHANNELS",
    "ConsoleApprovalChannel",
    "SlackApprovalChannel",
    "TelegramApprovalChannel",
    "build_approval_channel",
]
