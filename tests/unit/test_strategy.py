"""Unit tests for strategy v1: momentum signal, LLM sentiment, and floor_qty sizing.

All tests use fakes only — no DB, no network.
Each test maps to a named acceptance criterion from FIRM-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from firm.adapters.fakes import FakeLLM
from firm.agents.research import Claim, Evidence
from firm.config.settings import RiskPolicyConfig
from firm.domain import Bar, Portfolio
from firm.ports.types import LLMError, LLMMessage, LLMResponse
from firm.strategy import floor_qty
from firm.strategy.momentum import compute_momentum
from firm.strategy.sentiment import compute_sentiment

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TS = datetime(2024, 10, 22, 15, 0, 0, tzinfo=UTC)
_SYMBOL = "NVDA"


def _bar(close: Decimal, offset_days: int = 0) -> Bar:
    return Bar(
        symbol=_SYMBOL,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100_000,
        ts=_TS - timedelta(days=offset_days),
    )


def _rising_bars(n: int = 7) -> list[Bar]:
    """Return *n* bars with strictly increasing close prices."""
    return [_bar(Decimal(str(100 + i * 10)), offset_days=n - 1 - i) for i in range(n)]


def _falling_bars(n: int = 7) -> list[Bar]:
    """Return *n* bars with strictly decreasing close prices."""
    return [_bar(Decimal(str(160 - i * 10)), offset_days=n - 1 - i) for i in range(n)]


def _flat_bars(n: int = 7, close: Decimal = Decimal("150")) -> list[Bar]:
    """Return *n* bars with identical close prices."""
    return [_bar(close, offset_days=n - 1 - i) for i in range(n)]


def _risk_policy(
    max_trade_notional_pct: float = 0.10,
    buy_threshold: float = 0.1,
    sell_threshold: float = -0.1,
) -> RiskPolicyConfig:
    return RiskPolicyConfig(
        max_trade_notional_pct=max_trade_notional_pct,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=0.05,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        momentum_weight=0.6,
        sentiment_weight=0.4,
        momentum_lookback_days=5,
        max_events_per_symbol_per_hour=3,
        event_relevance_threshold=0.7,
        slippage_bps=5,
        commission_per_share=0.005,
        token_budget_per_cycle=50000,
    )


def _portfolio(cash: Decimal = Decimal("100_000")) -> Portfolio:
    return Portfolio(cash=cash)


def _evidence(claims_text: list[str] | None = None) -> Evidence:
    if claims_text is None:
        claims_text = ["NVDA beat earnings estimates."]
    return Evidence(
        symbol=_SYMBOL,
        claims=[
            Claim(text=t, source_url="https://example.com", chunk_id=f"c{i}")
            for i, t in enumerate(claims_text)
        ],
        retrieved_at=_TS,
    )


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=50,
        output_tokens=10,
        model="claude-haiku-4-5",
    )


# ---------------------------------------------------------------------------
# compute_momentum tests
# ---------------------------------------------------------------------------


class TestComputeMomentum:
    """Tests for ``firm.strategy.momentum.compute_momentum``."""

    def test_momentum_positive(self) -> None:
        """Rising bars → positive momentum float."""
        bars = _rising_bars(7)
        result = compute_momentum(bars, n_days=5)
        assert result > 0.0

    def test_momentum_negative(self) -> None:
        """Falling bars → negative momentum float."""
        bars = _falling_bars(7)
        result = compute_momentum(bars, n_days=5)
        assert result < 0.0

    def test_momentum_flat(self) -> None:
        """Unchanged close over all bars → momentum == 0.0."""
        bars = _flat_bars(7)
        result = compute_momentum(bars, n_days=5)
        assert result == 0.0

    def test_momentum_insufficient_data(self) -> None:
        """Fewer than n_days + 1 bars → ValueError."""
        bars = _rising_bars(4)  # 4 bars, n_days=5 requires 6
        with pytest.raises(ValueError, match="at least 6 bars"):
            compute_momentum(bars, n_days=5)

    def test_momentum_exact_boundary(self) -> None:
        """Exactly n_days + 1 bars → succeeds without error."""
        bars = _rising_bars(6)  # n_days=5 → minimum 6 bars
        result = compute_momentum(bars, n_days=5)
        assert isinstance(result, float)

    def test_momentum_uses_n_days_reference(self) -> None:
        """Return is computed as (close[-1] - close[-n_days]) / close[-n_days]."""
        bars = [
            _bar(Decimal("100"), offset_days=5),
            _bar(Decimal("110"), offset_days=4),
            _bar(Decimal("120"), offset_days=3),
            _bar(Decimal("130"), offset_days=2),
            _bar(Decimal("140"), offset_days=1),
            _bar(Decimal("150"), offset_days=0),
        ]
        # n_days=5 → bars[-5]=110, bars[-1]=150 → (150-110)/110 ≈ 0.3636
        result = compute_momentum(bars, n_days=5)
        expected = (150 - 110) / 110
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# compute_sentiment tests
# ---------------------------------------------------------------------------


class TestComputeSentiment:
    """Tests for ``firm.strategy.sentiment.compute_sentiment``."""

    def test_sentiment_clamped(self) -> None:
        """FakeLLM returns {"sentiment": 5.0} → clamped to 1.0."""
        llm = FakeLLM(responses=[_llm_response('{"sentiment": 5.0}')])
        ev = _evidence()
        result = compute_sentiment(ev, llm)
        assert result == 1.0

    def test_sentiment_negative_clamped(self) -> None:
        """FakeLLM returns {"sentiment": -9.0} → clamped to -1.0."""
        llm = FakeLLM(responses=[_llm_response('{"sentiment": -9.0}')])
        ev = _evidence()
        result = compute_sentiment(ev, llm)
        assert result == -1.0

    def test_sentiment_neutral_on_llm_error(self) -> None:
        """FakeLLM returns LLMError → fallback to 0.0 (neutral)."""

        @dataclass
        class ErrorLLM:
            def complete(
                self,
                messages: list[LLMMessage],
                *,
                model: str,
                max_tokens: int,
            ) -> LLMResponse | LLMError:
                return LLMError(message="rate limit", retryable=True)

            def count_tokens(
                self,
                messages: list[LLMMessage],
                *,
                model: str,
            ) -> int:
                return 0

        ev = _evidence()
        result = compute_sentiment(ev, ErrorLLM())
        assert result == 0.0

    def test_sentiment_neutral_on_json_parse_failure(self) -> None:
        """Malformed JSON response → fallback to 0.0."""
        llm = FakeLLM(responses=[_llm_response("not json at all")])
        ev = _evidence()
        result = compute_sentiment(ev, llm)
        assert result == 0.0

    def test_sentiment_valid_midrange(self) -> None:
        """LLM returns 0.6 → returned unchanged."""
        llm = FakeLLM(responses=[_llm_response('{"sentiment": 0.6}')])
        ev = _evidence()
        result = compute_sentiment(ev, llm)
        assert abs(result - 0.6) < 1e-9

    def test_sentiment_negative_midrange(self) -> None:
        """LLM returns -0.4 → returned unchanged."""
        llm = FakeLLM(responses=[_llm_response('{"sentiment": -0.4}')])
        ev = _evidence()
        result = compute_sentiment(ev, llm)
        assert abs(result - (-0.4)) < 1e-9


# ---------------------------------------------------------------------------
# floor_qty tests
# ---------------------------------------------------------------------------


class TestFloorQty:
    """Tests for ``firm.strategy.floor_qty`` (Portfolio-aware signature)."""

    def test_floor_qty_zero_on_weak_signal(self) -> None:
        """Very small signal near zero → qty may be 0 (no fractional shares)."""
        # signal=0.02, max_trade_notional_pct=0.10, NAV=100k → target=200
        # bar.close=1000 → 200/1000 = 0.2 → floor = 0 shares
        bar = Bar(
            symbol=_SYMBOL,
            open=Decimal("1000"),
            high=Decimal("1000"),
            low=Decimal("1000"),
            close=Decimal("1000"),
            volume=1_000,
            ts=_TS,
        )
        risk = _risk_policy(max_trade_notional_pct=0.10)
        portfolio = _portfolio(Decimal("100_000"))
        qty = floor_qty(0.02, portfolio, bar, risk)
        assert qty == Decimal("0")

    def test_floor_qty_respects_risk_cap(self) -> None:
        """Large signal → target notional capped by max_trade_notional_pct.

        The qty returned is the *floor* of (|signal| x max_pct x NAV) / close.
        With signal=1.0, max_pct=0.10, NAV=100k, close=100:
            target = 1.0 x 0.10 x 100,000 = 10,000
            qty = floor(10,000 / 100) = 100 shares
        """
        bar = Bar(
            symbol=_SYMBOL,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=10_000,
            ts=_TS,
        )
        risk = _risk_policy(max_trade_notional_pct=0.10)
        portfolio = _portfolio(Decimal("100_000"))
        qty = floor_qty(1.0, portfolio, bar, risk)
        # Expected: floor(1.0 x 0.10 x 100_000 / 100) = floor(100) = 100
        assert qty == Decimal("100")

    def test_floor_qty_zero_bar_close(self) -> None:
        """Bar with close=0 → qty is 0 (guard against division by zero)."""
        bar = Bar(
            symbol=_SYMBOL,
            open=Decimal("0"),
            high=Decimal("0"),
            low=Decimal("0"),
            close=Decimal("0"),
            volume=0,
            ts=_TS,
        )
        risk = _risk_policy()
        portfolio = _portfolio(Decimal("100_000"))
        qty = floor_qty(1.0, portfolio, bar, risk)
        assert qty == Decimal("0")

    def test_floor_qty_proportional_to_signal(self) -> None:
        """Higher signal magnitude → proportionally more shares (linear)."""
        bar = Bar(
            symbol=_SYMBOL,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=10_000,
            ts=_TS,
        )
        risk = _risk_policy(max_trade_notional_pct=0.10)
        portfolio = _portfolio(Decimal("100_000"))
        qty_half = floor_qty(0.5, portfolio, bar, risk)
        qty_full = floor_qty(1.0, portfolio, bar, risk)
        assert qty_half == Decimal("50")
        assert qty_full == Decimal("100")

    def test_floor_qty_never_negative(self) -> None:
        """Negative signal magnitude is absolute-valued; result is always ≥ 0."""
        bar = Bar(
            symbol=_SYMBOL,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=10_000,
            ts=_TS,
        )
        risk = _risk_policy(max_trade_notional_pct=0.10)
        portfolio = _portfolio(Decimal("100_000"))
        qty = floor_qty(-0.8, portfolio, bar, risk)
        assert qty >= Decimal("0")
