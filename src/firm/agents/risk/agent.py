"""RiskAgent — evaluate a TradeProposal against RiskPolicy limits."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

from firm.agents.base import BaseAgent
from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.agents.risk.schemas import ApprovedTrade, HITLRequired, Rejected, RiskInput
from firm.config.settings import RiskPolicyConfig
from firm.domain import (
    Approved,
    Portfolio,
    RiskPolicy,
    Trade,
    TradeStatus,
)
from firm.domain import HITLRequired as DomainHITLRequired


class RiskAgent(BaseAgent[RiskInput, ApprovedTrade | HITLRequired | Rejected]):
    def __init__(self, risk: RiskPolicyConfig) -> None:
        self._risk = risk

    def run(self, inp: RiskInput) -> ApprovedTrade | HITLRequired | Rejected:
        if isinstance(inp.proposal, Hold):
            return Rejected(
                reason=f"hold: {inp.proposal.reason}", correlation_id=inp.correlation_id
            )
        return _evaluate_proposal(
            inp.proposal, inp.portfolio, inp.prices, inp.correlation_id, self._risk
        )


def _build_domain_policy(config: RiskPolicyConfig) -> RiskPolicy:
    return RiskPolicy(
        max_trade_notional_pct=Decimal(str(config.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(config.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(config.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(config.hitl_threshold_pct)),
    )


def _build_trade_stub(
    proposal: TradeProposal,
    prices: dict[str, Decimal],
    correlation_id: str,
) -> Trade:
    price = prices.get(proposal.symbol, proposal.notional / proposal.qty)
    return Trade(
        id=uuid4(),
        cycle_id=_str_to_uuid(correlation_id),
        symbol=proposal.symbol,
        side=proposal.side,
        qty=proposal.qty,
        status=TradeStatus.PROPOSED,
        requested_price=price,
        idempotency_key=f"risk-check-{correlation_id}-{proposal.symbol}",
    )


def _str_to_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        return uuid4()


def _evaluate_proposal(
    proposal: TradeProposal,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    correlation_id: str,
    config: RiskPolicyConfig,
) -> ApprovedTrade | HITLRequired | Rejected:
    trade_stub = _build_trade_stub(proposal, prices, correlation_id)
    domain_policy = _build_domain_policy(config)
    result = domain_policy.check_trade(trade_stub, portfolio, prices)

    if isinstance(result, Approved):
        return ApprovedTrade(trade=trade_stub, correlation_id=correlation_id)
    if isinstance(result, DomainHITLRequired):
        return HITLRequired(
            proposal=proposal, reason=result.reason, correlation_id=correlation_id
        )
    return Rejected(reason=result.reason, correlation_id=correlation_id)
