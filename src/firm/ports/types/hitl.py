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

    model_config = {"frozen": True}


class ApprovalResult(BaseModel):
    status: ApprovalStatus
    decided_by: str | None = None
    edited_qty: Decimal | None = None

    model_config = {"frozen": True}
