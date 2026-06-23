"""Telegram approval channel — the blocking long-poll variant.

Adapts the existing :class:`firm.adapters.telegram.TelegramHITL` (which posts a
card and long-polls ``getUpdates`` until a callback) to the
:class:`firm.ports.approval.ApprovalChannel` Protocol. The two-step override UI
(approve / reject → buy|sell|hold) lives in the persistent ``firm bot``; here a
single tap resolves directly: approve → APPROVE, reject → OVERRIDE_HOLD, timeout
→ EXPIRE (fail-safe).
"""

from __future__ import annotations

from firm.adapters.telegram import TelegramHITL
from firm.domain.enums import HITLStatus
from firm.orchestration.hitl import HITLDecision
from firm.ports.types import HITLRequest

_STATUS_TO_DECISION: dict[HITLStatus, HITLDecision] = {
    HITLStatus.APPROVED: HITLDecision.APPROVE,
    HITLStatus.REJECTED: HITLDecision.OVERRIDE_HOLD,
}


class TelegramApprovalChannel:
    """Blocking Telegram approval surface backed by ``TelegramHITL``."""

    def __init__(self, hitl: TelegramHITL) -> None:
        self._hitl = hitl

    @classmethod
    def from_credentials(cls, token: str, chat_id: str) -> TelegramApprovalChannel:
        """Build from raw credentials (dry-run when absent/placeholder)."""
        return cls(TelegramHITL(token=token, chat_id=chat_id))

    def request_decision(self, request: HITLRequest) -> HITLDecision:
        result = self._hitl.send_hitl_request(request)
        # EXPIRED or any unmapped status → fail-safe timeout rejection.
        return _STATUS_TO_DECISION.get(result.status, HITLDecision.EXPIRE)
