"""Technical analysis agent I/O schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TechnicalInput(BaseModel):
    symbol: str
    decision_ts: datetime
    correlation_id: str

    model_config = {"frozen": True}


class TechnicalSignal(BaseModel):
    """Structured output from the TechnicalAnalysisAgent.

    The ``headline`` and ``body`` are LLM-written professional prose.
    Numeric indicators are computed deterministically from price bars.
    """

    symbol: str
    headline: str
    body: str
    bias: Literal["bullish", "bearish", "neutral"]
    rsi: float
    macd: float
    macd_cross: Literal["bullish", "bearish", "none"]
    bb_position: float  # 0.0 = price at lower band, 1.0 = at upper band
    key_support: float
    key_resistance: float

    model_config = {"frozen": True}


class TechnicalUnavailable(BaseModel):
    symbol: str
    reason: str

    model_config = {"frozen": True}
