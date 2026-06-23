"""ApprovalChannel port — the pluggable human-in-the-loop approval surface.

This is the *synchronous* (blocking) HITL seam used by the one-shot ``firm run``
command: present a trade-approval card to a human and block until they decide.
A channel takes a :class:`HITLRequest` and returns a structured
:class:`HITLDecision` (approve / override:buy / override:sell / override:hold /
expire) — it never resumes the graph itself, so the same decision flows through
the shared :func:`firm.orchestration.hitl.resume_decision` entry point regardless
of which channel produced it.

Implementations
---------------
- ``ConsoleApprovalChannel`` — interactive stdin prompt.
- ``TelegramApprovalChannel`` — wraps the blocking ``TelegramHITL`` long-poll.
- ``SlackApprovalChannel`` — sends the Block Kit card via ``SlackReportSink``;
  the *receive* side (Slack interactivity webhook) is the documented integration
  step (returns EXPIRED with a logged note until wired).

The persistent ``firm bot`` (Telegram long-poll) is the *asynchronous* variant:
it owns a single ``getUpdates`` consumer and resumes paused graph threads from a
callback handler, so it does not implement this blocking Protocol. It reuses the
shared formatters and ``resume_decision`` instead. New blocking channels
(email, SMS) only need to implement ``request_decision`` and register a factory.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from firm.orchestration.hitl import HITLDecision
from firm.ports.types import HITLRequest


@runtime_checkable
class ApprovalChannel(Protocol):
    """A synchronous human-in-the-loop approval surface.

    One method, honestly blocking: present the request and return the human's
    structured decision. On timeout an implementation MUST return
    ``HITLDecision.EXPIRE`` — never auto-approve (fail-safe).
    """

    def request_decision(self, request: HITLRequest) -> HITLDecision:
        """Send the approval card and block until the human decides.

        Returns the structured :class:`HITLDecision`. On timeout / undeliverable
        message, returns ``HITLDecision.EXPIRE`` (fail-safe — never auto-approve).
        """
        ...
