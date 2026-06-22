"""PortfolioManagerAgent — combine market momentum and news sentiment into a trade proposal."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from firm.agents.base import BaseAgent
from firm.agents.portfolio_manager.schemas import Hold, PMInput, TradeProposal
from firm.config.settings import RiskPolicyConfig
from firm.domain import Bar
from firm.domain.enums import TradeSide
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.strategy import compute_momentum, derive_sentiment, floor_qty, technical_score


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
        sentiment = derive_sentiment(inp.evidence, self._llm, inp.research_plan)
        signal = self._risk.momentum_weight * momentum + self._risk.sentiment_weight * sentiment + technical_score(inp.technical_signal)

        if signal > self._risk.buy_threshold:
            return _build_proposal(inp, signal, bar, momentum, sentiment, self._risk, TradeSide.BUY)
        if signal < self._risk.sell_threshold:
            return _build_proposal(inp, signal, bar, momentum, sentiment, self._risk, TradeSide.SELL)
        return Hold(symbol=inp.symbol, reason=f"signal={signal:.3f} in hold zone")


def _fetch_momentum(
    market_data: MarketDataSource,
    symbol: str,
    decision_ts: datetime,
    risk: RiskPolicyConfig,
) -> float:
    n_days = risk.momentum_lookback_days
    bars = market_data.get_bars(symbol, decision_ts - timedelta(days=n_days + 5), decision_ts)
    if len(bars) < n_days + 1:
        return 0.0
    try:
        return compute_momentum(bars, n_days)
    except ValueError:
        return 0.0


def _build_proposal(
    inp: PMInput,
    signal: float,
    bar: Bar,
    momentum: float,
    sentiment: float,
    risk: RiskPolicyConfig,
    side: TradeSide,
) -> TradeProposal | Hold:
    if side == TradeSide.SELL:
        holding = inp.portfolio.holdings.get(inp.symbol)
        if holding is None or holding.quantity <= Decimal("0"):
            return Hold(symbol=inp.symbol, reason="no position to sell")

    qty = floor_qty(signal, inp.portfolio, bar, risk)
    if side == TradeSide.SELL:
        holding = inp.portfolio.holdings.get(inp.symbol)
        if holding is not None:
            qty = min(qty, holding.quantity)
    if qty <= Decimal("0"):
        return Hold(symbol=inp.symbol, reason="sizing yielded zero quantity")

    return TradeProposal(
        symbol=inp.symbol,
        side=side,
        qty=qty,
        notional=qty * bar.close,
        rationale=_rationale(momentum, sentiment, signal, inp.technical_signal, inp.research_plan),
    )


def _rationale(momentum: float, sentiment: float, signal: float, technical: object, research_plan: object) -> str:
    from firm.agents.research_manager.schemas import ResearchPlan
    from firm.agents.technical.schemas import TechnicalSignal

    ta_part = f" ta_bias={technical.bias} rsi={technical.rsi:.1f}" if isinstance(technical, TechnicalSignal) else ""
    rp_part = f" debate={research_plan.recommendation}@{research_plan.conviction:.2f}" if isinstance(research_plan, ResearchPlan) else ""
    return f"momentum={momentum:.3f} sentiment={sentiment:.3f} signal={signal:.3f}{ta_part}{rp_part}"
