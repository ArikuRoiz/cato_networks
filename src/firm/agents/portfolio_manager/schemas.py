"""Portfolio manager I/O schemas.

``PortfolioManagerAgent`` has been dissolved — sizing is now handled by the
deterministic ``size_position`` tool in ``firm.tools.size_position``.
``PMInput`` has been removed (no callers remain after R1).

``TradeProposal`` and ``Hold`` are kept here because ``RiskAgent``,
``ExecutionAgent``, and the eval harness all import them from this package.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from firm.domain.enums import TradeSide


class TradeProposal(BaseModel):
    symbol: str
    side: TradeSide
    qty: Decimal
    notional: Decimal
    rationale: str

    model_config = {"frozen": True}


class Hold(BaseModel):
    """Decision not to trade — signal in hold zone or data unavailable."""

    symbol: str
    reason: str

    model_config = {"frozen": True}
