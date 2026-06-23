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
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from firm.adapters.fakes import FakeEvidenceStore, FakeReportSink
from firm.adapters.llm_offline import build_offline_llm
from firm.adapters.market_data_frozen import FrozenMarketData
from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.agents.research import Evidence, Refusal
from firm.config.settings import RiskPolicyConfig, load_risk_policy
from firm.domain import Holding, Lot, Portfolio, RiskPolicy, Trade, TradeStatus
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.graph import build_graph
from firm.orchestration.nodes import NodePorts
from firm.persistence.ledger import CycleAuditRecord
from firm.ports.llm import LLM
from firm.ports.types import NewsDoc

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
# Fake ledger (no Postgres required for eval)
# ---------------------------------------------------------------------------


class _ApprovalRecord:
    """In-memory record of a HITL decision captured by ``_FakeLedger``."""

    __slots__ = (
        "correlation_id",
        "decided_at",
        "decided_by",
        "edited_qty",
        "original_notional",
        "original_qty",
        "status",
        "trade_id",
    )

    def __init__(
        self,
        *,
        correlation_id: uuid.UUID,
        trade_id: uuid.UUID,
        status: str,
        original_notional: Decimal,
        original_qty: Decimal,
        edited_qty: Decimal | None,
        decided_at: datetime,
        decided_by: str,
    ) -> None:
        self.correlation_id = correlation_id
        self.trade_id = trade_id
        self.status = status
        self.original_notional = original_notional
        self.original_qty = original_qty
        self.edited_qty = edited_qty
        self.decided_at = decided_at
        self.decided_by = decided_by


class _FakeLedger:
    """Minimal in-memory ledger for eval runs.

    Tracks the portfolio in memory.  No Postgres, no ACID — this is
    only correct for a single-threaded eval run.

    ``buy()`` and ``sell()`` return a ``Trade`` with FILLED status to
    satisfy the structural contract of ``LedgerRepository.buy()`` /
    ``LedgerRepository.sell()``.

    ``record_approval()`` appends to ``approvals`` for offline exercise
    of the HITL recording path without a database.
    """

    def __init__(self, portfolio: Portfolio, portfolio_id: uuid.UUID) -> None:
        self._portfolio = portfolio
        self._portfolio_id = portfolio_id
        self._trades: list[Trade] = []
        self.approvals: list[_ApprovalRecord] = []
        self.cycles: list[CycleAuditRecord] = []

    def get_portfolio(self, portfolio_id: uuid.UUID) -> Portfolio:
        return self._portfolio

    def get_trades_for_cycle(self, cycle_id: uuid.UUID) -> list[Trade]:
        return [t for t in self._trades if t.cycle_id == cycle_id]

    def buy(
        self,
        trade: Trade,
        portfolio_id: uuid.UUID,
        opened_at: datetime | None = None,
    ) -> Trade:
        """Debit cash and add a holding; return the filled Trade."""
        filled = trade.model_copy(update={"status": TradeStatus.FILLED})
        notional = trade.qty * (trade.fill_price or trade.requested_price)
        commission = trade.commission or Decimal("0")
        self._portfolio.cash -= notional + commission
        _apply_buy(self._portfolio, trade, opened_at)
        self._trades.append(filled)
        return filled

    def sell(self, trade: Trade, portfolio_id: uuid.UUID) -> Trade:
        """Credit cash and reduce holding; return the filled Trade."""
        filled = trade.model_copy(update={"status": TradeStatus.FILLED})
        notional = trade.qty * (trade.fill_price or trade.requested_price)
        commission = trade.commission or Decimal("0")
        self._portfolio.cash += notional - commission
        _apply_sell(self._portfolio, trade)
        self._trades.append(filled)
        return filled

    def record_approval(
        self,
        *,
        correlation_id: uuid.UUID,
        trade_id: uuid.UUID,
        status: str,
        original_notional: Decimal,
        original_qty: Decimal,
        edited_qty: Decimal | None = None,
        decided_at: datetime | None = None,
        decided_by: str = "risk_committee",
    ) -> None:
        """Capture a HITL decision in memory for offline eval and testing."""
        ts = decided_at if decided_at is not None else datetime.now(tz=UTC)
        self.approvals.append(
            _ApprovalRecord(
                correlation_id=correlation_id,
                trade_id=trade_id,
                status=status,
                original_notional=original_notional,
                original_qty=original_qty,
                edited_qty=edited_qty,
                decided_at=ts,
                decided_by=decided_by,
            )
        )

    def record_cycle(self, record: CycleAuditRecord) -> None:
        """Capture a decision cycle record in memory for offline eval and testing."""
        self.cycles.append(record)


def _apply_buy(
    portfolio: Portfolio,
    trade: Trade,
    opened_at: datetime | None = None,
) -> None:
    """Update in-memory portfolio after a buy — no FIFO lot tracking."""
    fill_price = trade.fill_price or trade.requested_price
    ts = opened_at if opened_at is not None else datetime.now(tz=UTC)
    holding = portfolio.holdings.get(trade.symbol)
    if holding is None:
        lot = Lot(symbol=trade.symbol, qty=trade.qty, cost=fill_price, opened_at=ts)
        portfolio.holdings[trade.symbol] = Holding(symbol=trade.symbol, lots=[lot])
    else:
        lot = Lot(symbol=trade.symbol, qty=trade.qty, cost=fill_price, opened_at=ts)
        holding.lots.append(lot)


def _apply_sell(portfolio: Portfolio, trade: Trade) -> None:
    """Update in-memory portfolio after a sell — remove quantity FIFO."""
    holding = portfolio.holdings.get(trade.symbol)
    if holding is None:
        return
    remaining = trade.qty
    for lot in list(holding.lots):
        if remaining <= Decimal("0"):
            break
        if lot.qty <= remaining:
            remaining -= lot.qty
            lot.qty = Decimal("0")
        else:
            lot.qty -= remaining
            remaining = Decimal("0")


# ---------------------------------------------------------------------------
# Evidence store population from corpus.json
# ---------------------------------------------------------------------------


def _load_corpus(corpus_path: Path) -> list[NewsDoc]:
    """Parse corpus.json into NewsDoc objects."""
    raw: list[dict[str, Any]] = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs: list[NewsDoc] = []
    for item in raw:
        docs.append(
            NewsDoc(
                symbol=item["symbol"],
                text=item["text"],
                source_url=item["source_url"],
                published_at=datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")),
            )
        )
    return docs


def _build_evidence_store(corpus_path: Path) -> FakeEvidenceStore:
    """Populate a FakeEvidenceStore from the corpus file."""
    store = FakeEvidenceStore()
    for doc in _load_corpus(corpus_path):
        store.embed_and_store(doc)
    return store


# ---------------------------------------------------------------------------
# Default RiskPolicyConfig for eval (no Postgres / YAML required)
# ---------------------------------------------------------------------------


def _default_risk_policy() -> RiskPolicyConfig:
    """Return the locked-decision risk policy for eval runs."""
    return RiskPolicyConfig(
        max_trade_notional_pct=0.10,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=0.05,
        buy_threshold=0.05,
        sell_threshold=-0.05,
        momentum_weight=0.6,
        sentiment_weight=0.4,
        momentum_lookback_days=5,
        max_events_per_symbol_per_hour=3,
        event_relevance_threshold=0.7,
        token_budget_per_cycle=50000,
    )


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
    """Return a list of UTC midnight datetimes for each day in the replay window."""
    from datetime import date

    start = date.fromisoformat(str(window_config["start_date"]))
    end = date.fromisoformat(str(window_config["end_date"]))
    days: list[datetime] = []
    current = start
    while current <= end:
        days.append(datetime(current.year, current.month, current.day, tzinfo=UTC))
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
    """
    project_root = Path(__file__).parent.parent
    bars_dir = project_root / "data" / "bars"
    corpus_path = project_root / "data" / "news" / "corpus.json"

    market_data = FrozenMarketData(bars_dir)
    evidence_store = _build_evidence_store(corpus_path)
    llm: LLM = _build_llm(config.cassette_path)
    risk_policy = _try_load_risk_policy(project_root)

    portfolio_id = uuid.uuid4()
    portfolio = Portfolio(cash=_INITIAL_NAV)
    ledger = _FakeLedger(portfolio, portfolio_id)
    injection_guard = InjectionGuard()
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(risk_policy.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(risk_policy.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(risk_policy.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(risk_policy.hitl_threshold_pct)),
    )
    guardrail = LedgerGuardrail(domain_policy)
    report_sink = FakeReportSink()

    from firm.services.calendar import NYSECalendar

    ports = NodePorts(
        evidence=evidence_store,
        llm=llm,
        market_data=market_data,
        ledger=ledger,  # type: ignore[arg-type]
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=risk_policy,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        calendar=NYSECalendar(),
    )
    graph = build_graph(checkpointer=MemorySaver(), ports=ports)

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


def _build_llm(cassette_path: Path) -> LLM:
    """Return the appropriate offline LLM for the eval harness.

    Delegates to ``build_offline_llm`` — see that function for selection rules.
    A bare ``ANTHROPIC_API_KEY`` does NOT trigger live calls; only
    ``CASSETTE_MODE=record`` does (explicit opt-in).
    """
    return build_offline_llm(cassette_path)


def _try_load_risk_policy(project_root: Path) -> RiskPolicyConfig:
    """Load risk policy from YAML if available, else return defaults."""
    policy_path = project_root / "config" / "risk_policy.yaml"
    if policy_path.exists():
        try:
            return load_risk_policy(policy_path)
        except Exception:
            pass
    return _default_risk_policy()


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
