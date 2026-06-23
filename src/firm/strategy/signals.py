"""Pure signal computation functions: momentum and sentiment.

No IO imports — these are deterministic functions over domain values.
"""

from __future__ import annotations

import math
from decimal import Decimal

from firm.domain import Bar

# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PM signal helpers (moved here from portfolio_manager/agent.py)
# ---------------------------------------------------------------------------


def derive_sentiment(evidence: object, llm: object, research_plan: object = None) -> float:
    """Return a sentiment score in [-1, 1] from the best available source.

    Priority: ResearchPlan signal_score > LLM sentiment > keyword heuristic.
    """
    from firm.agents.research.schemas import Refusal
    from firm.agents.research_manager.schemas import ResearchPlan
    from firm.strategy.sentiment import compute_sentiment

    if isinstance(research_plan, ResearchPlan):
        return research_plan.signal_score
    if isinstance(evidence, Refusal):
        return 0.0
    from firm.agents.research.schemas import Evidence

    if llm is not None:
        return compute_sentiment(evidence, llm)  # type: ignore[arg-type]
    if isinstance(evidence, Evidence):
        return compute_sentiment_score([claim.text for claim in evidence.claims])
    return 0.0


def technical_score(technical: object) -> float:
    """Map TechnicalSignal bias to a [-0.3, 0.3] additive signal contribution."""
    from firm.agents.technical.schemas import TechnicalSignal
    from firm.domain.enums import TechnicalBias

    if not isinstance(technical, TechnicalSignal):
        return 0.0
    match technical.bias:
        case TechnicalBias.BULLISH:
            return 0.3
        case TechnicalBias.BEARISH:
            return -0.3
        case _:
            return 0.0
