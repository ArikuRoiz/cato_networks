"""Unit tests for the risk-node HITL resume / override-execution path.

Covers ``_resume_for_decision`` and its action helpers (``_override_buy``,
``_override_sell``, ``_proposal_from_approved``, ``_override_hold_proposal``)
using only the in-memory fakes — no Postgres, no LangGraph, no network.

The contract under test (see nodes.py):
  - approve        → executes the pre-sized recommended trade.
  - override:buy   → sizes a buy via ``size_position`` (conviction from
                     research_plan, else 0.5, capped by policy) and fills.
  - override:sell  → sells the existing holding (no-op → hold when none held).
  - override:hold  → no trade.
  - expire         → fail-safe rejection (rejected_timeout).

CRITICAL invariant: every executed override REWRITES ``state["trade_proposal"]``
to mirror the action that actually ran, so downstream synthesis / judge /
reporting never read a stale recommendation (an override-buy must not surface
as a "Hold" memo).
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from firm.adapters.fakes import (
    FakeCalendar,
    FakeEvidenceStore,
    FakeLLM,
    FakeMarketData,
    FakeReportSink,
)
from firm.agents.portfolio_manager.schemas import TradeProposal
from firm.agents.risk import ApprovedTrade, _build_trade_stub
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio, RiskPolicy
from firm.domain.enums import CycleOutcome, HITLStatus, TradeSide
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.domain.market import Bar
from firm.domain.portfolio import Holding, Lot
from firm.orchestration.hitl import HITLDecision
from firm.orchestration.nodes import (
    NodePorts,
    _override_buy,
    _override_sell,
    _resume_for_decision,
)
from firm.orchestration.state import GraphState

_DECISION_TS = "2024-10-23T10:00:00+00:00"
_PRICE = Decimal("200")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _policy() -> RiskPolicyConfig:
    return RiskPolicyConfig(
        max_trade_notional_pct=0.10,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=0.05,
        hitl_mode="always",
        buy_threshold=0.15,
        sell_threshold=-0.10,
        momentum_weight=0.60,
        sentiment_weight=0.40,
        momentum_lookback_days=5,
        max_events_per_symbol_per_hour=3,
        event_relevance_threshold=0.70,
        token_budget_per_cycle=50000,
    )


def _bar(symbol: str = "NVDA", close: Decimal = _PRICE) -> Bar:
    from datetime import UTC, datetime

    ts = datetime(2024, 10, 23, 10, 0, tzinfo=UTC)
    return Bar(symbol=symbol, open=close, high=close, low=close, close=close, volume=1000, ts=ts)


def _ports(portfolio: Portfolio, market_data: FakeMarketData) -> NodePorts:
    policy = _policy()
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(policy.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(policy.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(policy.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(policy.hitl_threshold_pct)),
    )
    return NodePorts(
        evidence=FakeEvidenceStore(),
        llm=FakeLLM(),
        market_data=market_data,
        ledger=None,  # type: ignore[arg-type]
        report_sink=FakeReportSink(),
        guardrail=LedgerGuardrail(domain_policy),
        injection_guard=InjectionGuard(),
        risk_policy=policy,
        portfolio_id=uuid.uuid4(),
        portfolio=portfolio,
        calendar=FakeCalendar(is_open=True),
    )


def _market_with_bar(symbol: str = "NVDA", close: Decimal = _PRICE) -> FakeMarketData:
    md = FakeMarketData()
    md.add_bar(_bar(symbol, close))
    return md


def _holding(symbol: str = "NVDA", qty: Decimal = Decimal("30")) -> Holding:
    from datetime import UTC, datetime

    lot = Lot(
        symbol=symbol,
        qty=qty,
        cost=Decimal("150"),
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return Holding(symbol=symbol, lots=[lot])


def _state(
    symbol: str = "NVDA",
    conviction: float | None = None,
    proposal_side: str = "buy",
) -> GraphState:
    research_plan = None
    if conviction is not None:
        research_plan = {"recommendation": "buy", "conviction": conviction}
    return GraphState(  # type: ignore[typeddict-item]
        correlation_id=str(uuid.uuid4()),
        trigger_type="scheduled",
        symbol=symbol,
        decision_ts=_DECISION_TS,
        evidence=None,
        research_plan=research_plan,
        trade_proposal={
            "symbol": symbol,
            "side": proposal_side,
            "qty": "10",
            "notional": "2000",
            "rationale": "desk recommendation",
        },
        approved_trade=None,
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )


def _recommended(symbol: str = "NVDA", side: TradeSide = TradeSide.BUY) -> ApprovedTrade:
    cid = str(uuid.uuid4())
    proposal = TradeProposal(
        symbol=symbol,
        side=side,
        qty=Decimal("10"),
        notional=Decimal("10") * _PRICE,
        rationale="desk recommendation",
    )
    trade = _build_trade_stub(proposal, {symbol: _PRICE}, cid)
    return ApprovedTrade(trade=trade, correlation_id=cid)


# ---------------------------------------------------------------------------
# approve — execute the recommended action
# ---------------------------------------------------------------------------


class TestApprove:
    def test_approve_executes_recommended_trade(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        recommended = _recommended()
        cid = recommended.correlation_id

        result = _resume_for_decision(ports, _state(), HITLDecision.APPROVE, recommended, cid)

        assert result["approved_trade"]["trade"]["qty"] == "10"
        assert result["hitl_decision"] == HITLDecision.APPROVE.value

    def test_approve_rewrites_trade_proposal_to_executed_action(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        recommended = _recommended(side=TradeSide.BUY)
        cid = recommended.correlation_id

        result = _resume_for_decision(ports, _state(), HITLDecision.APPROVE, recommended, cid)

        proposal = result["trade_proposal"]
        assert proposal["side"] == "buy"
        assert proposal["rationale"] == "hitl_approved:buy"


# ---------------------------------------------------------------------------
# override:buy
# ---------------------------------------------------------------------------


class TestOverrideBuy:
    def test_override_buy_sizes_and_fills(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        # No pre-recommended trade (e.g. desk said hold) — override buys anyway.
        result = _resume_for_decision(
            ports, _state(conviction=0.5), HITLDecision.OVERRIDE_BUY, None, str(uuid.uuid4())
        )

        trade = result["approved_trade"]["trade"]
        assert trade["side"] == "buy"
        assert Decimal(trade["qty"]) >= Decimal("1")
        assert result["hitl_decision"] == HITLDecision.OVERRIDE_BUY.value

    def test_override_buy_rewrites_trade_proposal_not_hold(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        # Desk proposal was a buy stub; the override must rewrite to the EXECUTED buy.
        result = _resume_for_decision(
            ports, _state(conviction=0.5), HITLDecision.OVERRIDE_BUY, None, str(uuid.uuid4())
        )

        proposal = result["trade_proposal"]
        assert proposal["side"] == "buy"
        assert proposal["rationale"] == "hitl_override:buy"
        # The executed qty must match the rewritten proposal qty (no stale "Hold").
        assert proposal["qty"] == result["approved_trade"]["trade"]["qty"]

    def test_override_buy_uses_conviction_from_plan(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        # conviction 1.0 → full cap; conviction 0.5 → half. Higher conviction → more shares.
        high = _override_buy(ports, _state(conviction=1.0), str(uuid.uuid4()))
        low = _override_buy(ports, _state(conviction=0.5), str(uuid.uuid4()))
        assert high is not None
        assert low is not None
        assert high.trade.qty > low.trade.qty

    def test_override_buy_defaults_conviction_when_no_plan(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        # No research_plan → conviction defaults to 0.5, still sizes a real trade.
        chosen = _override_buy(ports, _state(conviction=None), str(uuid.uuid4()))
        assert chosen is not None
        assert chosen.trade.qty >= Decimal("1")

    def test_override_buy_capped_by_policy(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        # conviction 1.0 → target = 1.0 * 10% * 100000 = 10000 → 10000/200 = 50 shares (the cap).
        chosen = _override_buy(ports, _state(conviction=1.0), str(uuid.uuid4()))
        assert chosen is not None
        assert chosen.trade.qty == Decimal("50")

    def test_override_buy_no_price_treated_as_hold(self) -> None:
        # Empty market data → no bar → returns None → resolves to hold.
        ports = _ports(Portfolio(cash=Decimal("100000")), FakeMarketData())
        result = _resume_for_decision(
            ports, _state(conviction=0.5), HITLDecision.OVERRIDE_BUY, None, str(uuid.uuid4())
        )
        assert result["cycle_outcome"] == CycleOutcome.HOLD
        assert "approved_trade" not in result


# ---------------------------------------------------------------------------
# override:sell
# ---------------------------------------------------------------------------


class TestOverrideSell:
    def test_override_sell_sells_existing_holding(self) -> None:
        portfolio = Portfolio(
            cash=Decimal("100000"), holdings={"NVDA": _holding(qty=Decimal("30"))}
        )
        ports = _ports(portfolio, _market_with_bar())

        result = _resume_for_decision(
            ports, _state(proposal_side="buy"), HITLDecision.OVERRIDE_SELL, None, str(uuid.uuid4())
        )

        trade = result["approved_trade"]["trade"]
        assert trade["side"] == "sell"
        assert Decimal(trade["qty"]) == Decimal("30")
        assert result["hitl_decision"] == HITLDecision.OVERRIDE_SELL.value

    def test_override_sell_rewrites_trade_proposal(self) -> None:
        portfolio = Portfolio(
            cash=Decimal("100000"), holdings={"NVDA": _holding(qty=Decimal("30"))}
        )
        ports = _ports(portfolio, _market_with_bar())

        result = _resume_for_decision(
            ports, _state(proposal_side="buy"), HITLDecision.OVERRIDE_SELL, None, str(uuid.uuid4())
        )
        proposal = result["trade_proposal"]
        assert proposal["side"] == "sell"
        assert proposal["rationale"] == "hitl_override:sell"

    def test_override_sell_no_holding_resolves_to_hold(self) -> None:
        # No position in NVDA → sell is a no-op → cycle holds.
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        result = _resume_for_decision(
            ports, _state(), HITLDecision.OVERRIDE_SELL, None, str(uuid.uuid4())
        )
        assert result["cycle_outcome"] == CycleOutcome.HOLD
        assert "approved_trade" not in result
        # The rewritten proposal must be a Hold (no side/qty) the operator effectively chose.
        proposal = result["trade_proposal"]
        assert "side" not in proposal
        assert proposal["symbol"] == "NVDA"
        assert "reason" in proposal

    def test_override_sell_returns_none_when_no_holding(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        assert _override_sell(ports, _state(), str(uuid.uuid4())) is None


# ---------------------------------------------------------------------------
# override:hold
# ---------------------------------------------------------------------------


class TestOverrideHold:
    def test_override_hold_executes_no_trade(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        result = _resume_for_decision(
            ports, _state(), HITLDecision.OVERRIDE_HOLD, _recommended(), str(uuid.uuid4())
        )
        assert result["cycle_outcome"] == CycleOutcome.HOLD
        assert "approved_trade" not in result
        assert result["hitl_decision"] == HITLDecision.OVERRIDE_HOLD.value

    def test_override_hold_rewrites_trade_proposal_to_hold(self) -> None:
        # Desk recommended a buy; the operator overrides to hold → proposal becomes a Hold,
        # so synthesis/judge describe a hold, not the stale buy.
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        result = _resume_for_decision(
            ports,
            _state(proposal_side="buy"),
            HITLDecision.OVERRIDE_HOLD,
            _recommended(),
            str(uuid.uuid4()),
        )
        proposal = result["trade_proposal"]
        # A Hold proposal carries no side/qty — just symbol + reason.
        assert "side" not in proposal
        assert "overrode the desk" in proposal["reason"]


# ---------------------------------------------------------------------------
# expire — fail-safe
# ---------------------------------------------------------------------------


class TestExpire:
    def test_expire_returns_rejected_timeout(self) -> None:
        ports = _ports(Portfolio(cash=Decimal("100000")), _market_with_bar())
        result = _resume_for_decision(
            ports, _state(), HITLDecision.EXPIRE, _recommended(), str(uuid.uuid4())
        )
        assert result["cycle_outcome"] == CycleOutcome.REJECTED_TIMEOUT
        assert result["hitl_decision"] == HITLDecision.EXPIRE.value
        assert "approved_trade" not in result


# ---------------------------------------------------------------------------
# hitl_status mapping (resume_decision contract)
# ---------------------------------------------------------------------------


class TestHITLStatusMapping:
    def test_overrides_map_to_approved(self) -> None:
        for decision in (
            HITLDecision.APPROVE,
            HITLDecision.OVERRIDE_BUY,
            HITLDecision.OVERRIDE_SELL,
            HITLDecision.OVERRIDE_HOLD,
        ):
            assert decision.hitl_status == HITLStatus.APPROVED

    def test_expire_maps_to_expired(self) -> None:
        assert HITLDecision.EXPIRE.hitl_status == HITLStatus.EXPIRED
