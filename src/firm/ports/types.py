"""Shared types used across port interfaces.

These are plain Pydantic models — no IO imports, no framework dependencies.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Evidence / RAG types
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrieved, scored document fragment from the evidence store."""

    id: UUID
    symbol: str
    text: str
    source_url: str
    chunk_id: str
    published_at: datetime
    score: float = 0.0
    embedding: list[float] = []

    model_config = {"frozen": True}

    @property
    def is_relevant(self) -> bool:
        """True when the retrieval score qualifies as a relevant event (>0.7)."""
        return self.score > 0.7


class NewsDoc(BaseModel):
    """A raw news document to be embedded and stored in the evidence store."""

    symbol: str
    text: str
    source_url: str
    published_at: datetime

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""

    role: Literal["user", "assistant", "system"]
    content: str

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    """A successful response from the LLM."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str

    model_config = {"frozen": True}


class LLMError(BaseModel):
    """A failed LLM call — retryable or terminal."""

    message: str
    retryable: bool

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# HITL (Human-in-the-loop) types
# ---------------------------------------------------------------------------


class HITLRequest(BaseModel):
    """A request sent to the human Risk Committee for trade approval."""

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
    """The outcome of a HITL approval request."""

    status: Literal["approved", "rejected", "edited", "expired"]
    decided_by: str | None = None
    edited_qty: Decimal | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Reporting types
# ---------------------------------------------------------------------------


class TradeRecord(TypedDict, total=False):
    """Shape of each entry in ``DailyReport.trades``."""

    cycle_id: str
    symbol: str
    side: str
    qty: float
    fill_price: float
    slippage: float
    commission: float
    status: str
    ts: str


class PositionRecord(TypedDict, total=False):
    """Shape of each entry in ``DailyReport.positions``."""

    symbol: str
    qty: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float


class CitationRecord(TypedDict, total=False):
    """Shape of each entry in ``DailyReport.citations``."""

    source_url: str
    chunk_id: str
    published_at: str
    symbol: str


class DailyReport(BaseModel):
    """Structured daily performance report for one trading date."""

    date: date
    nav: Decimal
    pnl: Decimal
    benchmark_return: float
    trades: list[TradeRecord]
    positions: list[PositionRecord]
    citations: list[CitationRecord]

    model_config = {"frozen": True}
