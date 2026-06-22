"""Pure signal computation functions: momentum and sentiment.

No IO imports — these are deterministic functions over domain values.
"""

from __future__ import annotations

import math
from decimal import Decimal

from firm.domain.entities import Bar

# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------


def compute_momentum_legacy(bars: list[Bar]) -> float:
    """Return the full-window price-return momentum signal, clamped to [-1, 1].

    Uses the (close[-1] - close[0]) / close[0] simple return over all bars
    in the provided window.  Returns 0.0 when fewer than two bars are given.

    .. deprecated::
        Use ``firm.strategy.momentum.compute_momentum`` instead.  This function
        is retained to avoid breaking any direct callers of ``signals.py``, but
        the name ``compute_momentum`` is no longer exported from this module to
        prevent shadowing the canonical implementation.
    """
    if len(bars) < 2:
        return 0.0
    oldest_close = float(bars[0].close)
    newest_close = float(bars[-1].close)
    if oldest_close == 0.0:
        return 0.0
    raw_return = (newest_close - oldest_close) / oldest_close
    return float(_clamp(raw_return, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Sentiment signal
# ---------------------------------------------------------------------------


def compute_sentiment_score(claims_texts: list[str]) -> float:
    """Derive a sentiment score in [-1, 1] from claim texts without an LLM call.

    Keyword-based heuristic: counts positive vs negative financial signals.
    This is intentionally simple; the LLM is used upstream for claim extraction
    only and must not emit numeric scores directly.
    """
    if not claims_texts:
        return 0.0
    pos_hits = sum(_positive_hit_count(t) for t in claims_texts)
    neg_hits = sum(_negative_hit_count(t) for t in claims_texts)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    raw = (pos_hits - neg_hits) / total
    return float(_clamp(raw, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Sizing helper
# ---------------------------------------------------------------------------


def floor_qty(
    signal: float,
    portfolio_nav: Decimal,
    bar_close: Decimal,
    max_trade_notional_pct: float,
) -> Decimal:
    """Compute the integer share quantity based on signal strength and NAV.

    The target notional is scaled by the absolute signal strength (max
    max_trade_notional_pct of NAV).  Returns 0 when the result is less than 1.
    """
    if bar_close <= Decimal("0"):
        return Decimal("0")
    target_notional = portfolio_nav * Decimal(str(abs(signal) * max_trade_notional_pct))
    qty = target_notional / bar_close
    floored = Decimal(str(math.floor(float(qty))))
    return max(Decimal("0"), floored)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_POSITIVE_KEYWORDS = (
    "beat",
    "exceed",
    "surpass",
    "strong",
    "growth",
    "revenue up",
    "raised guidance",
    "upgrade",
    "outperform",
    "record",
)
_NEGATIVE_KEYWORDS = (
    "miss",
    "below",
    "weak",
    "decline",
    "cut",
    "lowered guidance",
    "downgrade",
    "underperform",
    "loss",
    "disappoints",
)


def _positive_hit_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in _POSITIVE_KEYWORDS if kw in lower)


def _negative_hit_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lower)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
