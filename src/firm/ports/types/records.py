from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import NotRequired, TypedDict

from pydantic import BaseModel


class TradeRecord(TypedDict):
    cycle_id: str
    symbol: str
    side: str
    qty: float
    fill_price: float
    slippage: float
    commission: float
    status: str
    ts: NotRequired[str]


class PositionRecord(TypedDict):
    symbol: str
    qty: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float


class CitationRecord(TypedDict):
    source_url: str
    chunk_id: str
    published_at: str
    symbol: str


class DailyReport(BaseModel):
    date: date
    nav: Decimal
    pnl: Decimal
    benchmark_return: float
    trades: list[TradeRecord]
    positions: list[PositionRecord]
    citations: list[CitationRecord]

    model_config = {"frozen": True}
