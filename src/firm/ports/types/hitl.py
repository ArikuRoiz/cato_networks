"""Human-in-the-loop request and approval types."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from firm.domain.enums import ApprovalStatus


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

    @property
    def has_research_context(self) -> bool:
        """True when the research_plan fields have been populated."""
        return self.recommendation is not None

    @property
    def conviction_pct(self) -> str:
        """Conviction as a formatted percentage string, e.g. '100%'."""
        if self.conviction is None:
            return "-"
        return f"{self.conviction * 100:.0f}%"


class ApprovalResult(BaseModel):
    status: ApprovalStatus
    decided_by: str | None = None
    edited_qty: Decimal | None = None

    model_config = {"frozen": True}
