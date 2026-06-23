"""Deterministic position-sizing tool.

Takes the Research Manager's recommendation + conviction and converts them
into a whole-share quantity, capped by the per-trade RiskPolicy limit.

No LLM is involved — every number comes from domain objects and config.
"""

from __future__ import annotations

import math
from decimal import Decimal

from firm.domain.enums import Recommendation


def size_position(
    recommendation: Recommendation,
    conviction: float,
    nav: Decimal,
    price: Decimal,
    max_trade_notional_pct: float,
) -> Decimal:
    """Compute a whole-share quantity from a directional recommendation + conviction.

    Parameters
    ----------
    recommendation:
        Direction from the Research Manager (strong_buy…strong_sell).
        ``HOLD`` → returns 0.
    conviction:
        Confidence in [0, 1]. Scales target notional UP TO the per-trade cap.
    nav:
        Current portfolio NAV in dollars.
    price:
        Current share price in dollars.
    max_trade_notional_pct:
        Per-trade hard cap as a fraction of NAV (e.g. 0.10 = 10%).

    Returns
    -------
    Decimal
        Whole-share quantity ≥ 0.  Returns 0 when price ≤ 0, conviction ≤ 0,
        recommendation is HOLD, or the floored quantity is < 1.
    """
    if recommendation is Recommendation.HOLD:
        return Decimal("0")
    if price <= Decimal("0"):
        return Decimal("0")
    if conviction <= 0.0:
        return Decimal("0")

    target_notional = Decimal(str(conviction)) * Decimal(str(max_trade_notional_pct)) * nav
    raw_qty = target_notional / price
    floored = Decimal(str(math.floor(float(raw_qty))))
    return max(Decimal("0"), floored)


def trade_side_from_recommendation(recommendation: Recommendation) -> str | None:
    """Map a Recommendation to a TradeSide string, or None for HOLD.

    Returns ``"buy"`` for STRONG_BUY / BUY, ``"sell"`` for SELL / STRONG_SELL,
    and ``None`` for HOLD.
    """
    if recommendation in (Recommendation.STRONG_BUY, Recommendation.BUY):
        return "buy"
    if recommendation in (Recommendation.SELL, Recommendation.STRONG_SELL):
        return "sell"
    return None
