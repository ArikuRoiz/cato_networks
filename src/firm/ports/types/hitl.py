"""Human-in-the-loop request and approval types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from firm.domain.enums import ApprovalStatus

_DEFAULT_EXPIRY_MINUTES = 10


class HITLRequest(BaseModel):
    trade_id: UUID
    symbol: str
    side: str
    qty_str: str
    notional: Decimal
    reason: str
    expires_at: datetime
    correlation_id: str

    # Human-readable fields sourced from ResearchPlan — all optional so
    # existing callers (console HITL, tests) need no changes.
    recommendation: str | None = None
    conviction: float | None = None
    rationale: str | None = None
    bull_case: str | None = None
    bear_case: str | None = None

    model_config = {"frozen": True}

    @classmethod
    def from_interrupt(
        cls,
        interrupt_payload: dict[str, Any],
        correlation_id: str,
        expiry_minutes: int = _DEFAULT_EXPIRY_MINUTES,
    ) -> HITLRequest:
        """Build a request from a graph interrupt payload.

        Shared by every blocking approval channel and the async bot so the
        trade-summary + research-context extraction lives in exactly one place.
        """
        proposal = interrupt_payload.get("trade_proposal") or {}
        research_plan = interrupt_payload.get("research_plan") or {}
        return cls(
            trade_id=uuid.UUID(str(proposal.get("id", uuid.uuid4()))),
            symbol=str(proposal.get("symbol", "?")),
            side=str(proposal.get("side", "buy")),
            qty_str=str(proposal.get("qty", "0")),
            notional=Decimal(str(proposal.get("notional", "0"))),
            reason=str(proposal.get("rationale", "Risk Committee review required")),
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=expiry_minutes),
            correlation_id=correlation_id,
            recommendation=_opt_str(research_plan.get("recommendation")),
            conviction=_opt_float(research_plan.get("conviction")),
            rationale=_opt_str(research_plan.get("rationale")),
            bull_case=_opt_str(research_plan.get("bull_summary")),
            bear_case=_opt_str(research_plan.get("bear_summary")),
        )


class ApprovalResult(BaseModel):
    status: ApprovalStatus
    decided_by: str | None = None
    edited_qty: Decimal | None = None

    model_config = {"frozen": True}


def _opt_str(value: Any) -> str | None:
    """Coerce a present, truthy payload value to ``str``; else ``None``."""
    return str(value) if value else None


def _opt_float(value: Any) -> float | None:
    """Coerce a present payload value to ``float``; else ``None``."""
    return float(value) if value is not None else None
