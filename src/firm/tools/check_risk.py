"""Advisory risk-check tool.

Wraps ``RiskPolicy.check_trade`` so the sizing step can self-validate before
handing off to the mandatory risk gate at execution time.  This is advisory
only — the mandatory re-validation at execution stays unchanged (defense-in-depth).
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from firm.agents.portfolio_manager.schemas import TradeProposal
from firm.domain import Portfolio, RiskPolicy
from firm.domain.decisions import PolicyResult
from firm.domain.trade import Trade, TradeStatus


def check_risk(
    trade: TradeProposal,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    policy: RiskPolicy,
) -> PolicyResult:
    """Run an advisory risk check on *trade* against *policy*.

    Builds a minimal ``Trade`` domain object from the proposal and delegates to
    ``RiskPolicy.check_trade``.  The result is for informational use only;
    the authoritative gate lives in the execution node.

    Parameters
    ----------
    trade:
        The proposed trade to evaluate.
    portfolio:
        Current portfolio (for NAV and concentration checks).
    prices:
        Mark prices keyed by symbol.
    policy:
        The ``RiskPolicy`` to evaluate against.

    Returns
    -------
    PolicyResult
        ``Approved``, ``HITLRequired``, or ``Rejected`` from the domain layer.
    """
    price = prices.get(trade.symbol, _implied_price(trade))
    trade_stub = Trade(
        id=uuid4(),
        cycle_id=uuid4(),
        symbol=trade.symbol,
        side=trade.side,
        qty=trade.qty,
        status=TradeStatus.PROPOSED,
        requested_price=price,
        idempotency_key=f"advisory-{uuid4().hex}",
    )
    return policy.check_trade(trade_stub, portfolio, prices)


def _implied_price(trade: TradeProposal) -> Decimal:
    """Derive implied price from notional / qty, or 0 when qty is zero."""
    if trade.qty > Decimal("0"):
        return trade.notional / trade.qty
    return Decimal("0")
