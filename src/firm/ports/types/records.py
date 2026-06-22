"""Reporting record types (TypedDicts) and DailyReport."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TypedDict

from pydantic import BaseModel


class TradeRecord(TypedDict, total=False):
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
    symbol: str
    qty: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float


class CitationRecord(TypedDict, total=False):
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
