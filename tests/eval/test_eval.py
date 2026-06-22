"""Eval harness tests — reproducibility and honest negative-result reporting.

Tests:
- test_eval_reproducible: run twice with same cassette → identical EvalResult
- test_eval_reports_negative_alpha: simulate loss → report contains "underperformed"
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eval.metrics import compute_metrics
from eval.replay import CycleRecord, EvalConfig, EvalResult, run_eval
from eval.report import generate_report
from firm.domain import Bar

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_BARS_DIR = Path(__file__).parent.parent.parent / "data" / "bars"
_CORPUS_PATH = Path(__file__).parent.parent.parent / "data" / "news" / "corpus.json"


def _minimal_window_config() -> dict[str, Any]:
    """A minimal single-day window config for fast tests."""
    return {
        "name": "test_window",
        "start_date": "2024-10-21",
        "end_date": "2024-10-21",
        "benchmark": "SPY",
        "watchlist": ["NVDA"],
    }


def _eval_config(tmp_path: Path) -> EvalConfig:
    """EvalConfig wired to a non-existent cassette (FakeLLM path)."""
    return EvalConfig(
        window_config=_minimal_window_config(),
        cassette_path=tmp_path / "eval.jsonl",  # does not exist → FakeLLM
        output_dir=tmp_path / "output",
    )


def _spy_bars() -> list[Bar]:
    """Return two minimal SPY bars sufficient for benchmark computation."""
    return [
        Bar(
            symbol="SPY",
            open=Decimal("578.55"),
            high=Decimal("580.22"),
            low=Decimal("576.40"),
            close=Decimal("579.08"),
            volume=62_481_300,
            ts=datetime(2024, 10, 21, tzinfo=UTC),
        ),
        Bar(
            symbol="SPY",
            open=Decimal("583.40"),
            high=Decimal("585.15"),
            low=Decimal("581.90"),
            close=Decimal("584.10"),
            volume=58_913_600,
            ts=datetime(2024, 10, 25, tzinfo=UTC),
        ),
    ]


def _loss_result() -> EvalResult:
    """An EvalResult where the portfolio lost value (negative alpha scenario)."""
    cycle = CycleRecord(
        cycle_id=str(uuid.uuid4()),
        symbol="NVDA",
        decision_ts="2024-10-21T00:00:00+00:00",
        evidence_chunks=3,
        citations_used=3,
        has_citation=True,
        trade_proposed=True,
        trade_filled=True,
        tokens_used=200,
        llm_cost_usd=0.0001,
        guardrail_triggered=False,
        hitl_required=False,
        injection_detected=False,
        refusal=False,
        fill_price=140.15,
        fill_qty=10.0,
    )
    # NAV drops from 100k to 95k (-5 %) while SPY gains ~0.86 %
    return EvalResult(
        cycles=[cycle],
        portfolio_history=[
            {"date": "2024-10-21", "nav_usd": 100_000.0},
            {"date": "2024-10-25", "nav_usd": 95_000.0},
        ],
        initial_nav=100_000.0,
        final_nav=95_000.0,
    )


# ---------------------------------------------------------------------------
# test_eval_reproducible
# ---------------------------------------------------------------------------


def test_eval_reproducible(tmp_path: Path) -> None:
    """Two runs with the same (fake) cassette produce identical EvalResult.

    Reproducibility requires:
    - Identical cycle records (same symbols, same decisions)
    - Identical portfolio history NAV sequence
    - Identical initial/final NAV

    UUIDs (correlation_id, cycle_id) will differ between runs because they
    are generated fresh per run; we compare the structural/numeric fields.
    """
    config = _eval_config(tmp_path)

    result_a = run_eval(config)
    result_b = run_eval(config)

    assert len(result_a.cycles) == len(result_b.cycles), (
        "Both runs must produce the same number of cycles"
    )
    assert result_a.initial_nav == result_b.initial_nav
    assert result_a.final_nav == pytest.approx(result_b.final_nav, rel=1e-6)
    assert len(result_a.portfolio_history) == len(result_b.portfolio_history)

    for a_entry, b_entry in zip(
        result_a.portfolio_history, result_b.portfolio_history, strict=True
    ):
        assert a_entry["date"] == b_entry["date"]
        assert a_entry["nav_usd"] == pytest.approx(b_entry["nav_usd"], rel=1e-6)

    for a_cycle, b_cycle in zip(result_a.cycles, result_b.cycles, strict=True):
        assert a_cycle.symbol == b_cycle.symbol
        assert a_cycle.decision_ts == b_cycle.decision_ts
        assert a_cycle.evidence_chunks == b_cycle.evidence_chunks
        assert a_cycle.trade_proposed == b_cycle.trade_proposed
        assert a_cycle.trade_filled == b_cycle.trade_filled
        assert a_cycle.refusal == b_cycle.refusal
        assert a_cycle.guardrail_triggered == b_cycle.guardrail_triggered
        assert a_cycle.hitl_required == b_cycle.hitl_required
        assert a_cycle.tokens_used == b_cycle.tokens_used
        # Fill price and quantity must be identical (deterministic market data)
        if a_cycle.fill_price is not None:
            assert b_cycle.fill_price is not None, (
                "Run A has fill_price but run B does not — non-deterministic fill"
            )
            assert a_cycle.fill_price == pytest.approx(b_cycle.fill_price, rel=1e-9), (
                f"fill_price differs between runs: {a_cycle.fill_price} vs {b_cycle.fill_price}"
            )
        else:
            assert b_cycle.fill_price is None, (
                "Run B has fill_price but run A does not — non-deterministic fill"
            )
        if a_cycle.fill_qty is not None:
            assert b_cycle.fill_qty is not None
            assert a_cycle.fill_qty == pytest.approx(b_cycle.fill_qty, rel=1e-9), (
                f"fill_qty differs between runs: {a_cycle.fill_qty} vs {b_cycle.fill_qty}"
            )


# ---------------------------------------------------------------------------
# test_eval_reports_negative_alpha
# ---------------------------------------------------------------------------


def test_eval_reports_negative_alpha() -> None:
    """When the strategy loses money relative to the benchmark, the report
    states 'underperformed' plainly.

    This test constructs a synthetic EvalResult where the portfolio lost
    value while SPY gained, verifies compute_metrics yields negative alpha,
    and then asserts the generated report contains the word 'underperformed'.
    """
    result = _loss_result()
    spy_bars = _spy_bars()

    metrics = compute_metrics(result, spy_bars)

    # Sanity check: alpha must be negative for this scenario
    assert metrics.alpha < 0.0, (
        f"Expected negative alpha; got {metrics.alpha:.4%}. "
        f"total_return={metrics.total_return:.4%}, benchmark={metrics.benchmark_return:.4%}"
    )
    assert metrics.underperformed_benchmark

    report = generate_report(metrics, result)

    assert "underperformed" in report.lower(), (
        "Report must contain the word 'underperformed' when alpha < 0.\n"
        f"Report excerpt:\n{report[:500]}"
    )


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------


def test_metrics_total_return_zero_on_flat_nav() -> None:
    """Zero total return when initial_nav == final_nav."""
    result = EvalResult(
        cycles=[],
        portfolio_history=[{"date": "2024-10-21", "nav_usd": 100_000.0}],
        initial_nav=100_000.0,
        final_nav=100_000.0,
    )
    metrics = compute_metrics(result, _spy_bars())
    assert metrics.total_return == pytest.approx(0.0)


def test_metrics_groundedness_zero_when_no_citations() -> None:
    """groundedness_pct is 0.0 when no cycle has citations."""
    cycle = CycleRecord(
        cycle_id=str(uuid.uuid4()),
        symbol="NVDA",
        decision_ts="2024-10-21T00:00:00+00:00",
        evidence_chunks=0,
        citations_used=0,
        has_citation=False,
        trade_proposed=False,
        trade_filled=False,
        tokens_used=20,
        llm_cost_usd=0.0,
        guardrail_triggered=False,
        hitl_required=False,
        injection_detected=False,
        refusal=True,
        fill_price=None,
        fill_qty=None,
    )
    result = EvalResult(
        cycles=[cycle],
        portfolio_history=[{"date": "2024-10-21", "nav_usd": 100_000.0}],
        initial_nav=100_000.0,
        final_nav=100_000.0,
    )
    metrics = compute_metrics(result, _spy_bars())
    assert metrics.groundedness_pct == pytest.approx(0.0)
    assert metrics.refusal_rate == pytest.approx(100.0)


def test_metrics_guardrail_counted_correctly() -> None:
    """guardrail_trigger_count matches the number of triggered cycles."""

    def _cycle(guardrail: bool, hitl: bool) -> CycleRecord:
        return CycleRecord(
            cycle_id=str(uuid.uuid4()),
            symbol="AAPL",
            decision_ts="2024-10-21T00:00:00+00:00",
            evidence_chunks=2,
            citations_used=2,
            has_citation=True,
            trade_proposed=True,
            trade_filled=False,
            tokens_used=100,
            llm_cost_usd=0.00005,
            guardrail_triggered=guardrail,
            hitl_required=hitl,
            injection_detected=False,
            refusal=False,
            fill_price=None,
            fill_qty=None,
        )

    result = EvalResult(
        cycles=[
            _cycle(guardrail=True, hitl=False),
            _cycle(guardrail=False, hitl=True),
            _cycle(guardrail=True, hitl=True),
            _cycle(guardrail=False, hitl=False),
        ],
        portfolio_history=[{"date": "2024-10-21", "nav_usd": 100_000.0}],
        initial_nav=100_000.0,
        final_nav=100_000.0,
    )
    metrics = compute_metrics(result, _spy_bars())
    assert metrics.guardrail_trigger_count == 2
    assert metrics.hitl_count == 2


def test_report_positive_alpha_does_not_say_underperformed() -> None:
    """When alpha > 0 the report must NOT contain 'underperformed'."""
    # NAV grows from 100k to 110k (+10%) while SPY gains ~0.86%
    cycle = CycleRecord(
        cycle_id=str(uuid.uuid4()),
        symbol="NVDA",
        decision_ts="2024-10-21T00:00:00+00:00",
        evidence_chunks=3,
        citations_used=3,
        has_citation=True,
        trade_proposed=True,
        trade_filled=True,
        tokens_used=200,
        llm_cost_usd=0.0001,
        guardrail_triggered=False,
        hitl_required=False,
        injection_detected=False,
        refusal=False,
        fill_price=140.15,
        fill_qty=10.0,
    )
    result = EvalResult(
        cycles=[cycle],
        portfolio_history=[
            {"date": "2024-10-21", "nav_usd": 100_000.0},
            {"date": "2024-10-25", "nav_usd": 110_000.0},
        ],
        initial_nav=100_000.0,
        final_nav=110_000.0,
    )
    metrics = compute_metrics(result, _spy_bars())
    assert metrics.alpha > 0.0
    report = generate_report(metrics, result)
    assert "underperformed" not in report.lower()
    assert "outperformed" in report.lower()


def test_report_contains_required_sections() -> None:
    """The report must contain all required section headers."""
    result = _loss_result()
    metrics = compute_metrics(result, _spy_bars())
    report = generate_report(metrics, result)

    required_sections = [
        "## Return Summary",
        "## Process Metrics",
        "## Guardrail Events",
        "## Sample Trade",
        "## Limitations and Caveats",
    ]
    for section in required_sections:
        assert section in report, f"Missing section: {section!r}"


def test_run_eval_with_single_day_window(tmp_path: Path) -> None:
    """run_eval completes without error for a minimal single-day window."""
    config = _eval_config(tmp_path)
    result = run_eval(config)

    # One symbol (NVDA), one day → exactly one cycle record
    assert len(result.cycles) == 1
    assert result.cycles[0].symbol == "NVDA"
    assert result.initial_nav == pytest.approx(100_000.0)
    assert len(result.portfolio_history) == 1


def test_hallucination_check_fails_on_injection(tmp_path: Path) -> None:
    """hallucination_check_pass_rate < 100% when injection_detected is True."""
    cycle = CycleRecord(
        cycle_id=str(uuid.uuid4()),
        symbol="NVDA",
        decision_ts="2024-10-21T00:00:00+00:00",
        evidence_chunks=1,
        citations_used=0,
        has_citation=False,
        trade_proposed=False,
        trade_filled=False,
        tokens_used=50,
        llm_cost_usd=0.0,
        guardrail_triggered=False,
        hitl_required=False,
        injection_detected=True,  # <-- injection in corpus text
        refusal=True,
        fill_price=None,
        fill_qty=None,
    )
    result = EvalResult(
        cycles=[cycle],
        portfolio_history=[{"date": "2024-10-21", "nav_usd": 100_000.0}],
        initial_nav=100_000.0,
        final_nav=100_000.0,
    )
    metrics = compute_metrics(result, _spy_bars())
    assert metrics.hallucination_check_pass_rate < 100.0
