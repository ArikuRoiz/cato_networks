"""RiskPolicy — limit enforcement, returns a PolicyResult."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel

from firm.domain.decisions import Approved, HITLRequired, PolicyResult, Rejected

if TYPE_CHECKING:
    from firm.domain.portfolio import Portfolio
    from firm.domain.trade import Trade


class RiskPolicy(BaseModel):
    max_trade_notional_pct: Decimal
    max_name_concentration_pct: Decimal
    daily_loss_halt_pct: Decimal
    hitl_threshold_pct: Decimal

    model_config = {"frozen": True}

    def check_daily_halt(
        self,
        current_nav: Decimal,
        start_of_day_nav: Decimal,
    ) -> PolicyResult:
        """Return ``Rejected`` if intraday loss has breached the daily halt threshold."""
        if start_of_day_nav == Decimal("0"):
            return Rejected(reason="Start-of-day NAV is zero; cannot evaluate daily halt.")
        loss_pct = (start_of_day_nav - current_nav) / start_of_day_nav
        if loss_pct >= self.daily_loss_halt_pct:
            return Rejected(
                reason=(
                    f"Daily loss {loss_pct:.2%} has reached or exceeded halt "
                    f"threshold -{self.daily_loss_halt_pct:.2%}; trading halted."
                )
            )
        return Approved()

    def check_trade(
        self,
        trade: Trade,
        portfolio: Portfolio,
        prices: dict[str, Decimal],
        start_of_day_nav: Decimal | None = None,
    ) -> PolicyResult:
        """Evaluate all risk limits for *trade*."""
        nav = portfolio.nav(prices)
        if nav == Decimal("0"):
            return Rejected(reason="Portfolio NAV is zero; cannot size trade.")
        if start_of_day_nav is not None:
            halt = self.check_daily_halt(nav, start_of_day_nav)
            if isinstance(halt, Rejected):
                return halt
        notional = trade.qty * trade.requested_price
        return _evaluate_trade_limits(self, trade, notional, nav, portfolio, prices)


def _evaluate_trade_limits(
    policy: RiskPolicy,
    trade: Trade,
    notional: Decimal,
    nav: Decimal,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
) -> PolicyResult:
    trade_pct = notional / nav

    if trade_pct > policy.max_trade_notional_pct:
        return Rejected(
            reason=(
                f"Trade notional {trade_pct:.2%} exceeds max "
                f"{policy.max_trade_notional_pct:.2%} of NAV."
            )
        )

    if trade_pct > policy.hitl_threshold_pct:
        return HITLRequired(
            reason=(
                f"Trade notional {trade_pct:.2%} exceeds HITL "
                f"threshold {policy.hitl_threshold_pct:.2%} of NAV."
            )
        )

    if trade.side == "buy":
        projected = _projected_weight(trade, portfolio, prices, nav)
        if projected > policy.max_name_concentration_pct:
            return Rejected(
                reason=(
                    f"Post-trade weight {projected:.2%} exceeds max "
                    f"name concentration {policy.max_name_concentration_pct:.2%}."
                )
            )

    return Approved()


def _projected_weight(
    trade: Trade,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    nav: Decimal,
) -> Decimal:
    current_qty = Decimal("0")
    if trade.symbol in portfolio.holdings:
        current_qty = portfolio.holdings[trade.symbol].quantity
    new_qty = current_qty + trade.qty
    return new_qty * prices.get(trade.symbol, trade.requested_price) / nav
