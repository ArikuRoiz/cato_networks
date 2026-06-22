"""ReportSink port — the IO seam for external reporting and HITL approval.

Agents import this Protocol; adapters (Excel+Slack live, fake) implement it.
HITL timeout must auto-reject — ``send_hitl_request`` must never auto-approve
on expiry (fail-safe).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from firm.ports.types import ApprovalResult, DailyReport, HITLRequest


@runtime_checkable
class ReportSink(Protocol):
    """Outbound reporting and human-in-the-loop approval surface.

    Implementations must honour the ``runtime_checkable`` contract so fakes
    can be verified with ``isinstance`` in tests.
    """

    def send_daily_report(self, report: DailyReport) -> None:
        """Deliver the daily performance report via the configured channel."""
        ...

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Send a trade-approval request and block until the human decides.

        On timeout the adapter MUST return ``ApprovalResult(status="expired")``
        — never auto-approve.
        """
        ...

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Deliver an operational alert (circuit-breaker trip, guardrail breach, etc.)."""
        ...
