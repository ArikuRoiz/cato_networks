"""Technical indicator math — pure functions, no IO."""

from __future__ import annotations

from firm.domain import Bar


def compute_indicators(bars: list[Bar]) -> dict[str, float]:
    closes = [float(b.close) for b in bars]
    volumes = [int(b.volume) for b in bars]
    rsi = _rsi(closes)
    macd_line, signal_line = _macd(closes)
    upper, mid, lower = _bollinger(closes)
    price = closes[-1]
    bb_position = (price - lower) / (upper - lower) if upper != lower else 0.5
    return {
        "rsi": rsi,
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_position": bb_position,
        "avg_volume": sum(volumes[-10:]) / min(len(volumes), 10),
    }


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    avg_gain = sum(max(d, 0.0) for d in recent) / period
    avg_loss = sum(max(-d, 0.0) for d in recent) / period
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
    macd_line = _ema(closes[-26:], 12) - _ema(closes[-26:], 26)
    return macd_line, macd_line * 0.9  # 9-period signal approximation


def _bollinger(closes: list[float], period: int = 20) -> tuple[float, float, float]:
    window = closes[-period:] if len(closes) >= period else closes
    mid = sum(window) / len(window)
    std = (sum((x - mid) ** 2 for x in window) / len(window)) ** 0.5
    return mid + 2 * std, mid, mid - 2 * std
