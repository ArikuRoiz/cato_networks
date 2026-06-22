"""Market data value objects."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class Bar(BaseModel):
    """Single OHLCV candle, immutable."""

    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    ts: datetime

    model_config = {"frozen": True}
