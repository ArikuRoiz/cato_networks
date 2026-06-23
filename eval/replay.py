"""Historical replay harness.

Usage:
    python -m eval.replay --window data/windows/default.yaml

Wires FrozenMarketData + FakeEvidenceStore + CassetteLLM (or FakeLLM) into
the **production** graph (``firm.orchestration.build_graph``).  Each
(symbol, day) pair runs as an isolated cycle; results accumulate into an
``EvalResult`` for metric computation.

No live network calls are made.  The cassette must be pre-recorded or the
FakeLLM path is used for CI.

Design notes
------------
- The LangGraph checkpointer is an in-memory ``MemorySaver`` — no Postgres.
- The ledger is ``_FakeLedger`` — in-memory, no DB.
- Every number (fill price, NAV, portfolio history) comes from domain tools
  or market-data adapters, never from the LLM.
- Cassette misses in replay mode are caught by ``_GracefulLLM`` and returned
  as ``LLMError`` so agents fall back to their Failure variants and the cycle
  completes — the eval never crashes on a miss.
- HITL is auto-approved via the FakeReportSink (which returns APPROVED for
  every send_hitl_request call).  Trades that trigger the HITL interrupt in
  the risk_node will be resumed automatically by the eval runner.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from firm.adapters.fakes import FakeReportSink
from firm.adapters.market_data_frozen import FrozenMarketData
from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.agents.research import Evidence, Refusal

# Re-exported for tests that depend on the in-memory ledger living in eval.
# The implementation now lives in the composition root so the demo and eval
# share one fake ledger; ``_FakeLedger`` remains the public name here.
from firm.composition import InMemoryLedger as _FakeLedger
from firm.composition import build_offline_pipeline
from firm.domain import Portfolio

__all__ = ["CycleRecord", "EvalConfig", "EvalResult", "_FakeLedger", "run_eval"]

# ---------------------------------------------------------------------------
# Configuration and result schemas
# ---------------------------------------------------------------------------

_INITIAL_NAV: Decimal = Decimal("100000")  # $100k starting cash
_WATCHLIST: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"]


class EvalConfig(BaseModel):
    """Configuration for a single eval run."""

    window_config: dict[str, Any]
    cassette_path: Path
    output_dir: Path

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class CycleRecord(BaseModel):
    """Metrics captured for a single (symbol, day) decision cycle."""

    cycle_id: str
    symbol: str
    decision_ts: str

    # Evidence metrics
    evidence_chunks: int
    citations_used: int
    has_citation: bool

    # Trade lifecycle
    trade_proposed: bool
    trade_filled: bool

    # LLM cost
    tokens_used: int
    llm_cost_usd: float

    # Guardrail / safety
    guardrail_triggered: bool
    hitl_required: bool
    injection_detected: bool
    refusal: bool

    # Fill details (None when no trade was executed)
    fill_price: float | None
    fill_qty: float | None

    model_config = {"frozen": True}


class EvalResult(BaseModel):
    """Full result of a replay eval run."""

    cycles: list[CycleRecord]
    portfolio_history: list[dict[str, Any]]  # [{date, nav_usd}]
    initial_nav: float
    final_nav: float

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Single-cycle runner
# ---------------------------------------------------------------------------


def _run_cycle(
    symbol: str,
    decision_ts: datetime,
    correlation_id: str,
    graph: Any,
) -> CycleRecord:
    """Execute one full pipeline pass via the production LangGraph.

    Invokes the compiled graph with an in-memory thread and extracts
    CycleRecord fields from the final state.  Never raises — all failures
    are captured in the returned record.

    ``decision_ts`` is serialised to an ISO string before being placed in
    the initial state because the real graph's nodes parse it from a string
    (``_parse_datetime``).
    """
    thread_id = str(uuid.uuid4())
    initial_state = {
        "symbol": symbol,
        "decision_ts": decision_ts.isoformat(),
        "correlation_id": correlation_id,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state: dict[str, Any] = graph.invoke(initial_state, config=config)
        # always-HITL pauses the graph at the risk node every cycle. A backtest has
        # no human, so auto-approve the desk's recommendation and run to completion
        # (this is what makes synthesis/judge run and trades fill in the eval).
        if graph.get_state(config).next:
            from firm.orchestration.hitl import resume_decision

            final_state = resume_decision(graph, thread_id, "approve")
    except Exception:
        return _make_record(
            cycle_id=correlation_id,
            symbol=symbol,
            decision_ts=decision_ts,
            evidence_chunks=0,
            citations_used=0,
            has_citation=False,
            trade_proposed=False,
            trade_filled=False,
            tokens_used=0,
            guardrail_triggered=False,
            hitl_required=False,
            injection_detected=False,
            refusal=False,
            fill_price=None,
            fill_qty=None,
        )

    return _state_to_record(final_state, symbol, decision_ts, correlation_id)


def _state_to_record(
    state: dict[str, Any],
    symbol: str,
    decision_ts: datetime,
    correlation_id: str,
) -> CycleRecord:
    """Convert final GraphState dict to a CycleRecord.

    Reads from the real ``GraphState`` field names produced by the production
    graph nodes: ``evidence``, ``trade_proposal``, ``approved_trade``,
    ``cycle_outcome``.
    """
    evidence_raw: dict[str, Any] | None = state.get("evidence")
    trade_proposal_raw: dict[str, Any] | None = state.get("trade_proposal")
    approved_raw: dict[str, Any] | None = state.get("approved_trade")
    cycle_outcome: str | None = state.get("cycle_outcome")

    evidence = _deserialise_evidence(evidence_raw)
    trade_proposal = _deserialise_proposal(trade_proposal_raw) if trade_proposal_raw else None

    refusal = isinstance(evidence, Refusal)
    injection_detected = isinstance(evidence, Refusal) and evidence.reason == "injection_detected"

    evidence_chunks, citations_used, has_citation = _extract_evidence_metrics(evidence)
    trade_proposed = isinstance(trade_proposal, TradeProposal)
    trade_filled = cycle_outcome == "filled"
    guardrail_triggered = cycle_outcome in ("rejected", "error")
    hitl_required = _was_hitl_auto_approved(approved_raw, trade_proposed)

    fill_price, fill_qty = _extract_fill(approved_raw, trade_filled)
    tokens_used = _estimate_tokens(evidence) + _estimate_tokens(trade_proposal)

    return _make_record(
        cycle_id=correlation_id,
        symbol=symbol,
        decision_ts=decision_ts,
        evidence_chunks=evidence_chunks,
        citations_used=citations_used,
        has_citation=has_citation,
        trade_proposed=trade_proposed,
        trade_filled=trade_filled,
        tokens_used=tokens_used,
        guardrail_triggered=guardrail_triggered,
        hitl_required=hitl_required,
        injection_detected=injection_detected,
        refusal=refusal,
        fill_price=fill_price,
        fill_qty=fill_qty,
    )


def _extract_evidence_metrics(evidence: object) -> tuple[int, int, bool]:
    """Return (evidence_chunks, citations_used, has_citation) from a parsed evidence."""
    if not isinstance(evidence, Evidence):
        return 0, 0, False
    chunks = len(evidence.claims)
    cited = _count_cited_claims(evidence)
    return chunks, cited, cited > 0


def _extract_fill(
    approved_raw: dict[str, Any] | None,
    trade_filled: bool,
) -> tuple[float | None, float | None]:
    """Return (fill_price, fill_qty) when the cycle produced a fill."""
    if not trade_filled or approved_raw is None:
        return None, None
    trade_raw = approved_raw.get("trade", {})
    price_raw = trade_raw.get("requested_price") or trade_raw.get("fill_price")
    qty_raw = trade_raw.get("qty")
    fill_price = float(price_raw) if price_raw is not None else None
    fill_qty = float(qty_raw) if qty_raw is not None else None
    return fill_price, fill_qty


def _was_hitl_auto_approved(
    approved_raw: dict[str, Any] | None,
    trade_proposed: bool,
) -> bool:
    """Heuristic: HITL was triggered when there is an approved trade from a proposed trade.

    The real graph uses LangGraph interrupt() which does not leave a
    distinguishing flag in the final state; we approximate HITL by checking
    whether an approved_trade exists alongside a trade_proposed flag.
    In eval runs HITL is auto-resolved by the FakeReportSink.
    """
    return trade_proposed and approved_raw is not None


def _count_cited_claims(evidence: Evidence) -> int:
    """Count claims that have a non-empty source_url."""
    return sum(1 for c in evidence.claims if c.source_url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    cycle_id: str,
    symbol: str,
    decision_ts: datetime,
    evidence_chunks: int,
    citations_used: int,
    has_citation: bool,
    trade_proposed: bool,
    trade_filled: bool,
    tokens_used: int,
    guardrail_triggered: bool,
    hitl_required: bool,
    injection_detected: bool,
    refusal: bool,
    fill_price: float | None,
    fill_qty: float | None,
) -> CycleRecord:
    """Construct a CycleRecord with cost estimate."""
    cost = _estimate_cost(tokens_used)
    return CycleRecord(
        cycle_id=cycle_id,
        symbol=symbol,
        decision_ts=decision_ts.isoformat(),
        evidence_chunks=evidence_chunks,
        citations_used=citations_used,
        has_citation=has_citation,
        trade_proposed=trade_proposed,
        trade_filled=trade_filled,
        tokens_used=tokens_used,
        llm_cost_usd=cost,
        guardrail_triggered=guardrail_triggered,
        hitl_required=hitl_required,
        injection_detected=injection_detected,
        refusal=refusal,
        fill_price=fill_price,
        fill_qty=fill_qty,
    )


def _estimate_tokens(result: object) -> int:
    """Rough token estimate based on result type and content size."""
    if isinstance(result, Evidence):
        total = sum(len(c.text) for c in result.claims)
        return max(50, total // 4)
    if isinstance(result, (Refusal, Hold)):
        return 20
    if isinstance(result, TradeProposal):
        return len(result.rationale) // 4 + 30
    return 0


def _estimate_cost(tokens: int) -> float:
    """Haiku pricing: ~$0.25/1M input + $1.25/1M output (blended ~$0.50/1M)."""
    return tokens * 0.50 / 1_000_000


def _deserialise_evidence(raw: dict[str, Any] | None) -> Evidence | Refusal:
    """Deserialise evidence dict to Evidence or Refusal."""
    from firm.domain.enums import RefusalReason

    if raw is None:
        return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)
    if "claims" in raw:
        try:
            return Evidence.model_validate(raw)
        except Exception:
            pass
    if "reason" in raw:
        try:
            return Refusal.model_validate(raw)
        except Exception:
            pass
    return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)


def _deserialise_proposal(raw: dict[str, Any]) -> TradeProposal | Hold:
    """Deserialise proposal dict to TradeProposal or Hold."""
    if "qty" in raw and "notional" in raw:
        try:
            return TradeProposal.model_validate(raw)
        except Exception:
            pass
    if "reason" in raw:
        try:
            return Hold.model_validate(raw)
        except Exception:
            pass
    return Hold(symbol=raw.get("symbol", ""), reason="deserialisation failed")


def _parse_window_dates(window_config: dict[str, Any]) -> list[datetime]:
    """Return one UTC datetime per replay day, anchored at mid-session.

    Time-of-day matters: the execution node gates fills on NYSE market hours, so a
    midnight timestamp reads as "market closed" and every trade is rejected. We
    anchor each cycle at 16:00 UTC (≈ noon ET) so real trading days pass the gate.
    """
    from datetime import date

    start = date.fromisoformat(str(window_config["start_date"]))
    end = date.fromisoformat(str(window_config["end_date"]))
    days: list[datetime] = []
    current = start
    while current <= end:
        days.append(datetime(current.year, current.month, current.day, 16, 0, tzinfo=UTC))
        current += timedelta(days=1)
    return days


def _has_any_bar(
    day: datetime,
    market_data: FrozenMarketData,
    symbols: list[str],
) -> bool:
    """Return True when at least one symbol has a bar for *day*."""
    return any(market_data.get_bar(sym, day) is not None for sym in symbols)


def _snapshot_nav(
    portfolio: Portfolio,
    market_data: FrozenMarketData,
    symbols: list[str],
    day: datetime,
) -> float:
    """Compute NAV from portfolio + closing prices for *day*."""
    prices: dict[str, Decimal] = {}
    for sym in symbols:
        bar = market_data.get_bar(sym, day)
        if bar is not None:
            prices[sym] = bar.close
    held_symbols = set(portfolio.holdings.keys())
    filtered_prices = {s: p for s, p in prices.items() if s in held_symbols}
    equity = sum(
        portfolio.holdings[s].quantity * filtered_prices[s]
        for s in held_symbols
        if s in filtered_prices
    )
    return float(portfolio.cash + equity)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval(config: EvalConfig) -> EvalResult:
    """Run the full replay eval and return an EvalResult.

    Loads FrozenMarketData, populates FakeEvidenceStore from corpus.json,
    wires a graceful LLM (CassetteLLM wrapped to degrade on miss, or FakeLLM),
    builds NodePorts, compiles the **production** graph via
    ``firm.orchestration.build_graph``, then iterates over every (symbol, day)
    pair in the replay window.  Days with no bar data are skipped.

    The portfolio is shared across all cycles within the run — trades
    accumulate on the same in-memory portfolio, mirroring what a live run
    would see.

    The graph + ports + in-memory ledger are assembled by the shared
    composition root (:func:`firm.composition.build_offline_pipeline`); the
    eval supplies its auto-approving ``FakeReportSink`` and its own cassette
    path so a missing cassette falls back to the deterministic ``FakeLLM``.
    """
    project_root = Path(__file__).parent.parent
    market_data = FrozenMarketData(project_root / "data" / "bars")

    pipeline = build_offline_pipeline(
        project_root,
        initial_cash=_INITIAL_NAV,
        report_sink=FakeReportSink(),
        cassette_path=config.cassette_path,
    )
    graph = pipeline.graph
    portfolio = pipeline.portfolio

    watchlist: list[str] = list(config.window_config.get("watchlist", _WATCHLIST))
    days = _parse_window_dates(config.window_config)

    records: list[CycleRecord] = []
    portfolio_history: list[dict[str, Any]] = []

    for day in days:
        if not _has_any_bar(day, market_data, watchlist):
            continue

        for symbol in watchlist:
            correlation_id = str(uuid.uuid4())
            record = _run_cycle(
                symbol=symbol,
                decision_ts=day,
                correlation_id=correlation_id,
                graph=graph,
            )
            records.append(record)

        day_nav = _snapshot_nav(portfolio, market_data, watchlist, day)
        portfolio_history.append({"date": day.date().isoformat(), "nav_usd": day_nav})

    final_nav = portfolio_history[-1]["nav_usd"] if portfolio_history else float(_INITIAL_NAV)

    return EvalResult(
        cycles=records,
        portfolio_history=portfolio_history,
        initial_nav=float(_INITIAL_NAV),
        final_nav=final_nav,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a historical replay evaluation")
    parser.add_argument(
        "--window",
        required=True,
        metavar="PATH",
        help="Path to a replay-window YAML config (e.g. data/windows/default.yaml)",
    )
    parser.add_argument(
        "--cassette",
        metavar="PATH",
        default="data/cassettes/eval.jsonl",
        help="Path to the LLM cassette JSONL file (default: data/cassettes/eval.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="eval/output",
        help="Directory for eval output artifacts (default: eval/output)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: ``python -m eval.replay --window <path>``."""
    from eval.metrics import compute_metrics
    from eval.report import generate_report

    args = _parse_args()
    project_root = Path(__file__).parent.parent
    window_path = project_root / args.window

    if not window_path.exists():
        print(f"error: window config not found: {window_path}", file=sys.stderr)
        sys.exit(1)

    window_config = yaml.safe_load(window_path.read_text())
    cassette_path = project_root / args.cassette
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        window_config=window_config,
        cassette_path=cassette_path,
        output_dir=output_dir,
    )

    result = run_eval(config)

    bars_dir = project_root / "data" / "bars"
    market_data = FrozenMarketData(bars_dir)
    benchmark = window_config.get("benchmark", "SPY")
    days = _parse_window_dates(window_config)
    benchmark_bars = [
        b for day in days for b in [market_data.get_bar(benchmark, day)] if b is not None
    ]

    metrics = compute_metrics(result, benchmark_bars)
    report_md = generate_report(metrics, result)

    report_path = output_dir / "eval_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
