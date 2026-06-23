"""Slack approval channel — Block Kit send side wired, receive side stubbed.

The *send* side is real: it posts the HITL approval card via the existing
:class:`firm.adapters.report.SlackReportSink` Block Kit message (Approve / Reject
/ Edit buttons). The *receive* side — capturing the operator's button tap — needs
a Slack interactivity webhook (an HTTP endpoint Slack POSTs the action to,
correlated back by ``correlation_id``), which is an out-of-band async callback
this synchronous Protocol cannot block on.

Until that webhook is wired, ``request_decision`` posts the card and returns
``HITLDecision.EXPIRE`` (fail-safe — never auto-approve), logging a clear note.
Set ``raise_on_receive=True`` to surface the gap loudly during integration.

Extension points (no new infra in this repo):
- ``EMAIL``: an ``EmailApprovalChannel`` would send via SMTP and resolve replies
  through an inbound mail hook (or a magic-link approval URL) — same send-now /
  receive-out-of-band shape as Slack.
- ``SMS``: an ``SmsApprovalChannel`` over Twilio would send the summary and
  resolve via an inbound-SMS webhook keyed by ``correlation_id``.
Both follow this module's pattern: implement the send side against an existing
sink, mark the receive side as the integration step, and register a factory in
``registry.py``.
"""

from __future__ import annotations

import logging

from firm.adapters.report import SlackReportSink
from firm.orchestration.hitl import HITLDecision
from firm.ports.types import HITLRequest

logger = logging.getLogger(__name__)

_RECEIVE_NOTE = (
    "Slack receive needs a webhook: the operator's button tap arrives via a "
    "Slack interactivity webhook (out-of-band HTTP POST), which this blocking "
    "channel cannot wait on. Returning EXPIRED (fail-safe). Wire the webhook to "
    "resolve the decision by correlation_id=%s."
)


class SlackApprovalChannel:
    """Post the HITL card to Slack; receive side awaits webhook integration."""

    def __init__(self, sink: SlackReportSink, raise_on_receive: bool = False) -> None:
        self._sink = sink
        self._raise_on_receive = raise_on_receive

    @classmethod
    def from_channel(cls, channel: str, raise_on_receive: bool = False) -> SlackApprovalChannel:
        """Build from a Slack channel name (token resolved from env, dry-run otherwise)."""
        return cls(SlackReportSink(channel=channel), raise_on_receive=raise_on_receive)

    def request_decision(self, request: HITLRequest) -> HITLDecision:
        # Send side: real Block Kit card (or logged payload in dry-run).
        self._sink.send_hitl_request(request)

        if self._raise_on_receive:
            raise NotImplementedError(
                "SlackApprovalChannel receive side not wired: a Slack interactivity "
                "webhook is required to capture the operator's decision."
            )
        logger.warning(_RECEIVE_NOTE, request.correlation_id)
        return HITLDecision.EXPIRE
