"""PortfolioManagerAgent — combine market momentum and news sentiment into a trade proposal.

Input:  PMInput(symbol, evidence, portfolio, decision_ts, correlation_id)
Output: TradeProposal | Hold

Momentum is computed from market-data bars (no LLM).
Sentiment is computed by the LLM via ``compute_sentiment`` (structured JSON output
only — the LLM never emits a price, quantity, or P&L figure).
Sizing is performed by ``floor_qty`` using NAV and RiskPolicyConfig thresholds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from firm.agents.research import Evidence, Refusal
from firm.config.settings import RiskPolicyConfig
from firm.domain import Bar, Portfolio
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.strategy import compute_momentum, compute_sentiment, floor_qty
from firm.strategy.signals import compute_sentiment_score

# ---------------------------------------------------------------------------
# I/O schemas
# ---------------------------------------------------------------------------


class PMInput(BaseModel):
    """Input contract for PortfolioManagerAgent."""

    symbol: str
    evidence: Evidence | Refusal
    portfolio: Portfolio
    decision_ts: datetime
    correlation_id: str

    model_config = {"frozen": True}


class TradeProposal(BaseModel):
    """A sized, directional trade request ready for risk evaluation."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: Decimal
    notional: Decimal
    rationale: str

    model_config = {"frozen": True}


class Hold(BaseModel):
    """Decision not to trade — signal in hold zone or data unavailable."""

    symbol: str
    reason: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PortfolioManagerAgent:
    """Combine momentum and sentiment signals into a sized trade proposal."""

    def __init__(
        self,
        market_data: MarketDataSource,
        risk: RiskPolicyConfig,
        llm: LLM | None = None,
    ) -> None:
        self._market_data = market_data
        self._risk = risk
        self._llm = llm

    def run(self, inp: PMInput) -> TradeProposal | Hold:
        """Return a TradeProposal or Hold for the given symbol."""
        bar = self._market_data.get_bar(inp.symbol, inp.decision_ts)
        if bar is None:
            return Hold(symbol=inp.symbol, reason="no market data")

        momentum = _fetch_momentum(self._market_data, inp.symbol, inp.decision_ts, self._risk)
        sentiment = _derive_sentiment(inp.evidence, self._llm)
        signal = _combine_signal(momentum, sentiment, self._risk)

        if signal > self._risk.buy_threshold:
            return _build_buy_proposal(inp, signal, bar, momentum, sentiment, self._risk)
        if signal < self._risk.sell_threshold:
            return _build_sell_proposal(inp, signal, bar, momentum, sentiment, self._risk)
        return Hold(symbol=inp.symbol, reason=f"signal={signal:.3f} in hold zone")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_momentum(
    market_data: MarketDataSource,
    symbol: str,
    decision_ts: datetime,
    risk: RiskPolicyConfig,
) -> float:
    """Fetch bars for the momentum window and compute the signal.

    Uses the new ``compute_momentum(bars, n_days)`` from ``strategy.momentum``.
    Falls back to 0.0 when insufficient bars are available.
    """
    n_days = risk.momentum_lookback_days
    start = decision_ts - timedelta(days=n_days + 5)  # buffer for non-trading days
    bars = market_data.get_bars(symbol, start, decision_ts)
    if len(bars) < n_days + 1:
        return 0.0
    try:
        return compute_momentum(bars, n_days)
    except ValueError:
        return 0.0


def _derive_sentiment(evidence: Evidence | Refusal, llm: LLM | None) -> float:
    """Derive sentiment from Evidence; return 0.0 on Refusal or missing LLM.

    When an ``LLM`` port is available, delegates to ``compute_sentiment`` for
    structured JSON-based scoring.  Falls back to the keyword heuristic when
    no LLM is injected (e.g. during tests that do not need LLM sentiment).
    The LLM never emits a price, quantity, or date.
    """
    if isinstance(evidence, Refusal):
        return 0.0
    if llm is not None:
        return compute_sentiment(evidence, llm)
    texts = [claim.text for claim in evidence.claims]
    return compute_sentiment_score(texts)


def _combine_signal(momentum: float, sentiment: float, risk: RiskPolicyConfig) -> float:
    """Blend momentum and sentiment using configured weights."""
    return risk.momentum_weight * momentum + risk.sentiment_weight * sentiment


def _build_buy_proposal(
    inp: PMInput,
    signal: float,
    bar: Bar,
    momentum: float,
    sentiment: float,
    risk: RiskPolicyConfig,
) -> TradeProposal | Hold:
    """Build a buy proposal; return Hold when sizing yields zero shares."""
    qty = floor_qty(signal, inp.portfolio, bar, risk)
    if qty <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="sizing yielded zero quantity")
    notional = qty * bar.close
    return TradeProposal(
        symbol=inp.symbol,
        side="buy",
        qty=qty,
        notional=notional,
        rationale=f"momentum={momentum:.3f} sentiment={sentiment:.3f} signal={signal:.3f}",
    )


def _build_sell_proposal(
    inp: PMInput,
    signal: float,
    bar: Bar,
    momentum: float,
    sentiment: float,
    risk: RiskPolicyConfig,
) -> TradeProposal | Hold:
    """Build a sell proposal; return Hold when no position or zero sizing."""
    holding = inp.portfolio.holdings.get(inp.symbol)
    if holding is None or holding.quantity <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="no position to sell")
    qty = floor_qty(signal, inp.portfolio, bar, risk)
    qty = min(qty, holding.quantity)
    if qty <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="sizing yielded zero quantity")
    notional = qty * bar.close
    return TradeProposal(
        symbol=inp.symbol,
        side="sell",
        qty=qty,
        notional=notional,
        rationale=f"momentum={momentum:.3f} sentiment={sentiment:.3f} signal={signal:.3f}",
    )
