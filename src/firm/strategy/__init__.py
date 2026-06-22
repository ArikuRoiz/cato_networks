"""Momentum and news-sentiment signal computation (pure functions, no IO).

Public API
----------
- ``compute_momentum`` — N-day price-return from ``momentum.py``
- ``compute_sentiment`` — LLM-backed sentiment score from ``sentiment.py``
- ``floor_qty`` — NAV-aware integer share sizing (new Portfolio-aware signature)
- ``compute_sentiment_score`` — keyword heuristic from ``signals.py`` (legacy, no LLM)
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from firm.config.settings import RiskPolicyConfig
from firm.domain import Bar, Portfolio
from firm.strategy.momentum import compute_momentum
from firm.strategy.sentiment import compute_sentiment
from firm.strategy.signals import compute_sentiment_score


def floor_qty(
    signal: float,
    portfolio: Portfolio,
    bar: Bar,
    risk: RiskPolicyConfig,
) -> Decimal:
    """Compute integer share quantity from signal strength, NAV, and risk limits.

    The target notional is ``|signal| x max_trade_notional_pct x NAV``.  The
    result is floored to a whole number of shares and clamped to ``≥ 0``.

    The LLM never contributes to this calculation — all inputs come from
    domain objects and config.

    Parameters
    ----------
    signal:
        Combined momentum + sentiment signal (arbitrary float).
    portfolio:
        Current portfolio; used to compute NAV.
    bar:
        Current OHLCV bar for the symbol being sized.
    risk:
        ``RiskPolicyConfig`` containing ``max_trade_notional_pct``.

    Returns
    -------
    Decimal
        Whole-share quantity ``≥ 0``.
    """
    if bar.close <= Decimal("0"):
        return Decimal("0")
    nav = _portfolio_nav(portfolio, bar)
    target_notional = Decimal(str(abs(signal))) * Decimal(str(risk.max_trade_notional_pct)) * nav
    qty = (target_notional / bar.close).quantize(Decimal("1"), rounding=ROUND_DOWN)
    return max(qty, Decimal("0"))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _portfolio_nav(portfolio: Portfolio, bar: Bar) -> Decimal:
    """Compute NAV using the current bar close for the active symbol.

    Provides avg_cost as the mark for any other open holdings so that NAV
    is never silently undervalued.  Callers receive a correct NAV or a
    propagated domain error — no silent fallback.
    """
    prices: dict[str, Decimal] = {bar.symbol: bar.close}
    for sym, holding in portfolio.holdings.items():
        if sym != bar.symbol and holding.quantity > Decimal("0"):
            prices[sym] = holding.avg_cost
    return portfolio.nav(prices)


__all__ = [
    "compute_momentum",
    "compute_sentiment",
    "compute_sentiment_score",
    "floor_qty",
]
