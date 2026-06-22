"""TechnicalAnalysisAgent — tool-using agent that fetches price data and generates analysis.

The LLM calls ``get_price_and_indicators`` to retrieve OHLCV bars and computed
RSI/MACD/Bollinger values, then produces a structured JSON signal.  The LLM
controls what lookback window it requests rather than receiving a fixed dataset.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from firm.agents.base import BaseAgent
from firm.agents.technical.schemas import (
    TechnicalInput,
    TechnicalSignal,
    TechnicalUnavailable,
)
from firm.domain.enums import MACDCross, TechnicalBias
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.ports.types import LLMError, LLMMessage, ToolDef
from firm.strategy import compute_indicators
from firm.utils import parse_json_dict

_SYSTEM_PROMPT = (
    "You are a senior quantitative analyst. "
    "Use the get_price_and_indicators tool to retrieve market data, then "
    "respond ONLY with valid JSON — no markdown fences, no extra text:\n"
    '{"headline":"<one sentence ≤80 chars>","body":"<2-3 sentence analysis>",'
    '"bias":"bullish"|"bearish"|"neutral","key_support":<float>,"key_resistance":<float>}'
)

_PRICE_TOOL = ToolDef(
    name="get_price_and_indicators",
    description="Retrieve recent OHLCV price bars and computed technical indicators (RSI, MACD, Bollinger Bands) for the symbol.",
    input_schema={
        "type": "object",
        "properties": {
            "lookback_days": {
                "type": "integer",
                "description": "Number of calendar days of history to fetch (default 40, max 90)",
                "default": 40,
            },
        },
        "required": [],
    },
)


class TechnicalAnalysisAgent(BaseAgent[TechnicalInput, TechnicalSignal | TechnicalUnavailable]):
    def __init__(self, market_data: MarketDataSource, llm: LLM) -> None:
        self._market_data = market_data
        self._llm = llm

    def run(self, inp: TechnicalInput) -> TechnicalSignal | TechnicalUnavailable:
        indicators_snapshot: dict[str, float] = {}
        last_close: float = 0.0

        def get_price_and_indicators(args: dict[str, Any]) -> str:
            nonlocal indicators_snapshot, last_close
            lookback = min(int(args.get("lookback_days", 40)), 90)
            start = inp.decision_ts - timedelta(days=lookback + 5)
            bars = self._market_data.get_bars(inp.symbol, start, inp.decision_ts)
            if len(bars) < 14:
                return f"Insufficient price history for {inp.symbol}: only {len(bars)} bars available."
            ind = compute_indicators(bars)
            indicators_snapshot = ind
            last_close = float(bars[-1].close)
            rsi_label = "overbought" if ind["rsi"] > 70 else "oversold" if ind["rsi"] < 30 else "neutral"
            macd_label = "bullish" if ind["histogram"] > 0 else "bearish"
            return (
                f"Technical indicators for {inp.symbol} (close {last_close:.2f}):\n"
                f"- RSI(14): {ind['rsi']:.1f} ({rsi_label})\n"
                f"- MACD: {ind['macd']:.4f} | Signal: {ind['signal']:.4f} | "
                f"Histogram: {ind['histogram']:.4f} ({macd_label} momentum)\n"
                f"- Bollinger Bands (20,2): upper={ind['bb_upper']:.2f} | "
                f"mid={ind['bb_mid']:.2f} | lower={ind['bb_lower']:.2f}\n"
                f"- BB position: {ind['bb_position']:.1%} (0%=lower band, 100%=upper band)\n"
                f"- 10-day avg volume: {ind['avg_volume']:,.0f}"
            )

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"Analyze the technical picture for {inp.symbol} as of {inp.decision_ts.date()}.",
            ),
        ]

        resp = self._llm.complete_with_tools(
            messages,
            tools=[_PRICE_TOOL],
            executors={"get_price_and_indicators": get_price_and_indicators},
            model="haiku",
            max_tokens=512,
            max_rounds=3,
        )

        if isinstance(resp, LLMError):
            return TechnicalUnavailable(symbol=inp.symbol, reason=f"llm_error: {resp.message}")

        if not indicators_snapshot:
            return TechnicalUnavailable(symbol=inp.symbol, reason="insufficient price history")

        return _parse_signal(inp.symbol, resp.content, indicators_snapshot, last_close)


def _parse_signal(symbol: str, content: str, ind: dict[str, float], close: float) -> TechnicalSignal | TechnicalUnavailable:
    raw = parse_json_dict(content)
    if raw is None:
        return TechnicalUnavailable(symbol=symbol, reason="llm returned invalid JSON")

    if ind["histogram"] > 0.001:
        macd_cross = MACDCross.BULLISH
    elif ind["histogram"] < -0.001:
        macd_cross = MACDCross.BEARISH
    else:
        macd_cross = MACDCross.NONE

    return TechnicalSignal(
        symbol=symbol,
        headline=str(raw.get("headline", "Technical analysis unavailable"))[:120],
        body=str(raw.get("body", "")),
        bias=TechnicalBias(raw["bias"]) if raw.get("bias") in {b.value for b in TechnicalBias} else TechnicalBias.NEUTRAL,
        rsi=round(ind["rsi"], 2),
        macd=round(ind["macd"], 4),
        macd_cross=macd_cross,
        bb_position=round(ind["bb_position"], 3),
        key_support=float(raw.get("key_support", round(close * 0.97, 2))),
        key_resistance=float(raw.get("key_resistance", round(close * 1.03, 2))),
    )
