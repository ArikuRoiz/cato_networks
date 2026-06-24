"""Unit tests for the deterministic tools layer.

Covers size_position.  No LLM, no DB, no network.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from firm.domain.enums import Recommendation
from firm.tools.size_position import size_position, trade_side_from_recommendation

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NAV = Decimal("100_000")
_PRICE = Decimal("500")
_MAX_PCT = 0.10  # 10% per-trade cap


# ---------------------------------------------------------------------------
# size_position — direction → qty mapping
# ---------------------------------------------------------------------------


class TestSizePositionDirections:
    """Direction mapping: buy/sell recommendations → non-zero qty; hold → 0."""

    def test_strong_buy_yields_nonzero_qty(self) -> None:
        qty = size_position(
            recommendation=Recommendation.STRONG_BUY,
            conviction=0.8,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty >= Decimal("1"), "STRONG_BUY with 80% conviction should produce shares"

    def test_buy_yields_nonzero_qty(self) -> None:
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.5,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty >= Decimal("1"), "BUY with 50% conviction should produce shares"

    def test_hold_yields_zero(self) -> None:
        qty = size_position(
            recommendation=Recommendation.HOLD,
            conviction=1.0,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0"), "HOLD must never produce shares regardless of conviction"

    def test_sell_yields_nonzero_qty(self) -> None:
        qty = size_position(
            recommendation=Recommendation.SELL,
            conviction=0.5,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty >= Decimal("1"), "SELL with 50% conviction should produce shares"

    def test_strong_sell_yields_nonzero_qty(self) -> None:
        qty = size_position(
            recommendation=Recommendation.STRONG_SELL,
            conviction=0.9,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty >= Decimal("1"), "STRONG_SELL with 90% conviction should produce shares"


# ---------------------------------------------------------------------------
# size_position — conviction scaling
# ---------------------------------------------------------------------------


class TestSizePositionConviction:
    """Conviction scales quantity monotonically; zero/low conviction → 0."""

    def test_zero_conviction_yields_zero(self) -> None:
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.0,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0"), "conviction=0 must yield qty=0"

    def test_negative_conviction_yields_zero(self) -> None:
        """Negative conviction is treated as zero (should not happen, but guard it)."""
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=-0.5,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0")

    def test_higher_conviction_yields_more_shares(self) -> None:
        low = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.2,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        high = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.9,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert high >= low, "Higher conviction must not produce fewer shares"

    def test_full_conviction_respects_cap(self) -> None:
        """Even at conviction=1.0, notional must not exceed max_trade_notional_pct x NAV."""
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=1.0,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        notional = qty * _PRICE
        cap = _NAV * Decimal(str(_MAX_PCT))
        assert notional <= cap, f"notional {notional} must not exceed cap {cap}"


# ---------------------------------------------------------------------------
# size_position — edge cases
# ---------------------------------------------------------------------------


class TestSizePositionEdgeCases:
    """Edge cases: zero/negative price, very small NAV, whole-share floor."""

    def test_zero_price_yields_zero(self) -> None:
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.9,
            nav=_NAV,
            price=Decimal("0"),
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0"), "price=0 must yield qty=0 (division guard)"

    def test_negative_price_yields_zero(self) -> None:
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.9,
            nav=_NAV,
            price=Decimal("-100"),
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0"), "negative price must yield qty=0"

    def test_result_is_whole_shares(self) -> None:
        """Fractional shares must be floored to a whole number."""
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.77,
            nav=_NAV,
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        # qty must be an integer-valued Decimal
        assert qty == qty.to_integral_value(), f"qty {qty} must be a whole number"

    def test_tiny_nav_yields_zero_when_below_one_share(self) -> None:
        """When NAV is too small to buy even 1 share, qty must be 0."""
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=0.1,
            nav=Decimal("10"),  # $10 NAV, $500 price — cannot afford 1 share
            price=_PRICE,
            max_trade_notional_pct=_MAX_PCT,
        )
        assert qty == Decimal("0"), "qty < 1 must floor to 0"


# ---------------------------------------------------------------------------
# size_position — notional formula verification
# ---------------------------------------------------------------------------


class TestSizePositionFormula:
    """Verify the sizing formula: target = conviction x max_pct x NAV, floored."""

    @pytest.mark.parametrize(
        ("conviction", "nav", "price", "max_pct", "expected_qty"),
        [
            # conviction=1.0, nav=100k, price=500, cap=10% → target=10k, qty=20
            (1.0, Decimal("100000"), Decimal("500"), 0.10, Decimal("20")),
            # conviction=0.5, nav=100k, price=500, cap=10% → target=5k, qty=10
            (0.5, Decimal("100000"), Decimal("500"), 0.10, Decimal("10")),
            # conviction=0.3, nav=100k, price=1000, cap=10% → target=3k, qty=3
            (0.3, Decimal("100000"), Decimal("1000"), 0.10, Decimal("3")),
        ],
    )
    def test_formula(
        self,
        conviction: float,
        nav: Decimal,
        price: Decimal,
        max_pct: float,
        expected_qty: Decimal,
    ) -> None:
        qty = size_position(
            recommendation=Recommendation.BUY,
            conviction=conviction,
            nav=nav,
            price=price,
            max_trade_notional_pct=max_pct,
        )
        assert qty == expected_qty, (
            f"conviction={conviction} nav={nav} price={price} max_pct={max_pct}: "
            f"expected {expected_qty}, got {qty}"
        )


# ---------------------------------------------------------------------------
# trade_side_from_recommendation
# ---------------------------------------------------------------------------


class TestTradeSideFromRecommendation:
    """trade_side_from_recommendation maps enum values to string sides."""

    def test_strong_buy_is_buy(self) -> None:
        assert trade_side_from_recommendation(Recommendation.STRONG_BUY) == "buy"

    def test_buy_is_buy(self) -> None:
        assert trade_side_from_recommendation(Recommendation.BUY) == "buy"

    def test_hold_is_none(self) -> None:
        assert trade_side_from_recommendation(Recommendation.HOLD) is None

    def test_sell_is_sell(self) -> None:
        assert trade_side_from_recommendation(Recommendation.SELL) == "sell"

    def test_strong_sell_is_sell(self) -> None:
        assert trade_side_from_recommendation(Recommendation.STRONG_SELL) == "sell"

