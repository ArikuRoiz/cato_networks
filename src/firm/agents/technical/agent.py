"""TechnicalAnalysisAgent — compute RSI/MACD/Bollinger and generate an LLM narrative."""

from __future__ import annotations

import json
from datetime import timedelta

from firm.agents.base import BaseAgent
from firm.agents.technical.schemas import (
    TechnicalInput,
    TechnicalSignal,
    TechnicalUnavailable,
)
from firm.domain import Bar
from firm.ports.llm import LLM, LLMError
from firm.ports.market_data import MarketDataSource
from firm.ports.types import LLMMessage

_LOOKBACK_DAYS = 35  # enough for MACD(26) + buffer
_SYSTEM_PROMPT = (
    "You are a senior quantitative analyst writing for an internal trading report. "
    "Respond ONLY with valid JSON, no markdown fences, no extra text."
)


class TechnicalAnalysisAgent(BaseAgent[TechnicalInput, TechnicalSignal | TechnicalUnavailable]):
    def __init__(self, market_data: MarketDataSource, llm: LLM) -> None:
        self._market_data = market_data
        self._llm = llm

    def run(self, inp: TechnicalInput) -> TechnicalSignal | TechnicalUnavailable:
        start = inp.decision_ts - timedelta(days=_LOOKBACK_DAYS + 5)
        bars = self._market_data.get_bars(inp.symbol, start, inp.decision_ts)
        if len(bars) < 14:
            return TechnicalUnavailable(symbol=inp.symbol, reason="insufficient price history")

        indicators = _compute_indicators(bars)
        messages = _build_messages(inp.symbol, indicators, bars[-1])
        resp = self._llm.complete(messages, model="haiku", max_tokens=512)

        if isinstance(resp, LLMError):
            return TechnicalUnavailable(symbol=inp.symbol, reason=f"llm_error: {resp.message}")

        return _parse_signal(inp.symbol, resp.content, indicators, bars[-1])


# ---------------------------------------------------------------------------
# Indicator math (pure, no side effects)
# ---------------------------------------------------------------------------


def _compute_indicators(bars: list[Bar]) -> dict[str, float]:
    closes = [float(b.close) for b in bars]
    volumes = [int(b.volume) for b in bars]

    rsi = _rsi(closes)
    macd_line, signal_line = _macd(closes)
    upper, mid, lower = _bollinger(closes)
    price = closes[-1]

    bb_position = (price - lower) / (upper - lower) if upper != lower else 0.5
    avg_vol = sum(volumes[-10:]) / min(len(volumes), 10)

    return {
        "rsi": rsi,
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_position": bb_position,
        "avg_volume": avg_vol,
    }


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [max(d, 0.0) for d in recent]
    losses = [max(-d, 0.0) for d in recent]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0.0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def _macd(closes: list[float]) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    ema12 = _ema(closes[-26:], 12)
    ema26 = _ema(closes[-26:], 26)
    macd_line = ema12 - ema26
    signal_line = macd_line * 0.9  # 9-period approximation for small datasets
    return macd_line, signal_line


def _bollinger(closes: list[float], period: int = 20) -> tuple[float, float, float]:
    window = closes[-period:] if len(closes) >= period else closes
    mid = sum(window) / len(window)
    variance = sum((x - mid) ** 2 for x in window) / len(window)
    std = variance ** 0.5
    return mid + 2 * std, mid, mid - 2 * std


# ---------------------------------------------------------------------------
# LLM prompt + response parsing
# ---------------------------------------------------------------------------


def _build_messages(
    symbol: str,
    ind: dict[str, float],
    bar: Bar,
) -> list[LLMMessage]:
    rsi_label = "overbought" if ind["rsi"] > 70 else "oversold" if ind["rsi"] < 30 else "neutral"
    macd_cross = "bullish" if ind["histogram"] > 0 else "bearish"

    user = (
        f"Technical indicators for {symbol} (closing price {float(bar.close):.2f}):\n"
        f"- RSI(14): {ind['rsi']:.1f} ({rsi_label})\n"
        f"- MACD: {ind['macd']:.4f} | Signal: {ind['signal']:.4f} | "
        f"Histogram: {ind['histogram']:.4f} ({macd_cross} momentum)\n"
        f"- Bollinger Bands (20,2): upper={ind['bb_upper']:.2f} | "
        f"mid={ind['bb_mid']:.2f} | lower={ind['bb_lower']:.2f}\n"
        f"- BB position: {ind['bb_position']:.1%} (0%=lower band, 100%=upper band)\n"
        f"- 10-day avg volume: {ind['avg_volume']:,.0f}\n\n"
        "Respond ONLY with this JSON (no markdown):\n"
        '{"headline":"<one sentence ≤80 chars>","body":"<2-3 sentence professional analysis>",'
        '"bias":"bullish"|"bearish"|"neutral","key_support":<float>,"key_resistance":<float>}'
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    ]


def _parse_signal(
    symbol: str,
    content: str,
    ind: dict[str, float],
    bar: Bar,
) -> TechnicalSignal | TechnicalUnavailable:
    try:
        raw = json.loads(content.strip())
    except json.JSONDecodeError:
        return TechnicalUnavailable(symbol=symbol, reason="llm returned invalid JSON")

    macd_cross: str
    if ind["histogram"] > 0.001:
        macd_cross = "bullish"
    elif ind["histogram"] < -0.001:
        macd_cross = "bearish"
    else:
        macd_cross = "none"

    price = float(bar.close)
    return TechnicalSignal(
        symbol=symbol,
        headline=str(raw.get("headline", "Technical analysis unavailable"))[:120],
        body=str(raw.get("body", "")),
        bias=raw.get("bias", "neutral") if raw.get("bias") in ("bullish", "bearish", "neutral") else "neutral",
        rsi=round(ind["rsi"], 2),
        macd=round(ind["macd"], 4),
        macd_cross=macd_cross,  # type: ignore[arg-type]
        bb_position=round(ind["bb_position"], 3),
        key_support=float(raw.get("key_support", round(price * 0.97, 2))),
        key_resistance=float(raw.get("key_resistance", round(price * 1.03, 2))),
    )
