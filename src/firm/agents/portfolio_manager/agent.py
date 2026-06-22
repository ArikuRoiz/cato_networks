"""PortfolioManagerAgent — combine market momentum and news sentiment into a trade proposal."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from firm.agents.base import BaseAgent
from firm.agents.portfolio_manager.schemas import Hold, PMInput, TradeProposal
from firm.agents.research.schemas import Evidence, Refusal
from firm.config.settings import RiskPolicyConfig
from firm.domain import Bar
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.strategy import compute_momentum, compute_sentiment, floor_qty
from firm.strategy.signals import compute_sentiment_score


class PortfolioManagerAgent(BaseAgent[PMInput, TradeProposal | Hold]):
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


def _fetch_momentum(
    market_data: MarketDataSource,
    symbol: str,
    decision_ts: datetime,
    risk: RiskPolicyConfig,
) -> float:
    n_days = risk.momentum_lookback_days
    start = decision_ts - timedelta(days=n_days + 5)
    bars = market_data.get_bars(symbol, start, decision_ts)
    if len(bars) < n_days + 1:
        return 0.0
    try:
        return compute_momentum(bars, n_days)
    except ValueError:
        return 0.0


def _derive_sentiment(evidence: Evidence | Refusal, llm: LLM | None) -> float:
    if isinstance(evidence, Refusal):
        return 0.0
    if llm is not None:
        return compute_sentiment(evidence, llm)
    texts = [claim.text for claim in evidence.claims]
    return compute_sentiment_score(texts)


def _combine_signal(momentum: float, sentiment: float, risk: RiskPolicyConfig) -> float:
    return risk.momentum_weight * momentum + risk.sentiment_weight * sentiment


def _build_buy_proposal(
    inp: PMInput,
    signal: float,
    bar: Bar,
    momentum: float,
    sentiment: float,
    risk: RiskPolicyConfig,
) -> TradeProposal | Hold:
    qty = floor_qty(signal, inp.portfolio, bar, risk)
    if qty <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="sizing yielded zero quantity")
    return TradeProposal(
        symbol=inp.symbol,
        side="buy",
        qty=qty,
        notional=qty * bar.close,
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
    holding = inp.portfolio.holdings.get(inp.symbol)
    if holding is None or holding.quantity <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="no position to sell")
    qty = min(floor_qty(signal, inp.portfolio, bar, risk), holding.quantity)
    if qty <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="sizing yielded zero quantity")
    return TradeProposal(
        symbol=inp.symbol,
        side="sell",
        qty=qty,
        notional=qty * bar.close,
        rationale=f"momentum={momentum:.3f} sentiment={sentiment:.3f} signal={signal:.3f}",
    )
