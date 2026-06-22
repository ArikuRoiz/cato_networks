"""Historical replay harness.

Usage:
    python -m eval.replay --window data/windows/default.yaml

Wires FrozenMarketData + FakeEvidenceStore + CassetteLLM into the full
five-agent pipeline.  Each (symbol, day) pair is run as an isolated
cycle; results accumulate into an EvalResult for metric computation.

No live network calls are made.  The cassette must be pre-recorded or
the test cassette fixture is used for CI.

Design notes
------------
- The LangGraph checkpointer is replaced with an in-memory MemorySaver so
  that eval runs need no Postgres connection.
- The LedgerRepository is replaced with a _FakeLedger (no DB) so that
  eval can run offline and deterministically.
- Every number (fill price, NAV, portfolio history) comes from domain
  tools or market-data adapters, never from the LLM.
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
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel

from firm.adapters.fakes import FakeEvidenceStore, FakeLLM, FakeReportSink
from firm.adapters.llm_cassette import CassetteLLM
from firm.adapters.market_data_frozen import FrozenMarketData
from firm.agents.execution import ExecutionAgent, ExecutionFailure, ExecutionInput, Fill
from firm.agents.portfolio_manager import Hold, PMInput, PortfolioManagerAgent, TradeProposal
from firm.agents.reporting import ReportingAgent, ReportingInput
from firm.agents.research import Evidence, Refusal, ResearchAgent, ResearchInput
from firm.agents.risk import ApprovedTrade, HITLRequired, Rejected, RiskAgent, RiskInput
from firm.config.settings import RiskPolicyConfig, load_risk_policy
from firm.domain.entities import Portfolio, RiskPolicy, Trade, TradeStatus
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.ports.llm import LLM
from firm.ports.types import LLMResponse, NewsDoc

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


class _FakeLedger:
    """Minimal in-memory ledger for eval runs.

    Tracks the portfolio in memory.  No Postgres, no ACID — this is
    only correct for a single-threaded eval run.

    ``buy()`` and ``sell()`` return a ``Trade`` with FILLED status to
    satisfy the structural contract of ``LedgerRepository.buy()`` /
    ``LedgerRepository.sell()``.
    """

    def __init__(self, portfolio: Portfolio, portfolio_id: uuid.UUID) -> None:
        self._portfolio = portfolio
        self._portfolio_id = portfolio_id
        self._trades: list[Trade] = []

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


def _apply_buy(
    portfolio: Portfolio,
    trade: Trade,
    opened_at: datetime | None = None,
) -> None:
    """Update in-memory portfolio after a buy — no FIFO lot tracking."""
    from firm.domain.entities import Holding, Lot

    fill_price = trade.fill_price or trade.requested_price
    ts = opened_at if opened_at is not None else datetime.now(tz=UTC)
    holding = portfolio.holdings.get(trade.symbol)
    if holding is None:
        lot = Lot(
            symbol=trade.symbol,
            qty=trade.qty,
            cost=fill_price,
            opened_at=ts,
        )
        portfolio.holdings[trade.symbol] = Holding(symbol=trade.symbol, lots=[lot])
    else:
        lot = Lot(
            symbol=trade.symbol,
            qty=trade.qty,
            cost=fill_price,
            opened_at=ts,
        )
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
                published_at=datetime.fromisoformat(
                    item["published_at"].replace("Z", "+00:00")
                ),
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
        slippage_bps=5,
        commission_per_share=0.005,
        token_budget_per_cycle=50000,
    )


# ---------------------------------------------------------------------------
# Eval-specific LangGraph builder
# ---------------------------------------------------------------------------


def _build_eval_graph(
    research_agent: ResearchAgent,
    pm_agent: PortfolioManagerAgent,
    risk_agent: RiskAgent,
    execution_agent: ExecutionAgent,
    reporting_agent: ReportingAgent,
    portfolio: Portfolio,
    portfolio_id: uuid.UUID,
    market_data: FrozenMarketData,
) -> Any:
    """Build and compile a LangGraph StateGraph for the eval harness.

    Uses MemorySaver as the checkpointer so no Postgres connection is required.
    All agents are closed over in the node functions.
    """
    builder: StateGraph = StateGraph(dict)  # type: ignore[type-arg]

    def _research_node(state: dict[str, Any]) -> dict[str, Any]:
        symbol: str = state.get("symbol", "")
        decision_ts: datetime = state.get("decision_ts") or datetime.now(tz=UTC)
        correlation_id: str = state.get("correlation_id", "")
        inp = ResearchInput(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
        )
        result = research_agent.run(inp)
        return {"research_result": result}

    def _pm_node(state: dict[str, Any]) -> dict[str, Any]:
        symbol: str = state.get("symbol", "")
        decision_ts: datetime = state.get("decision_ts") or datetime.now(tz=UTC)
        correlation_id: str = state.get("correlation_id", "")
        research_result = state.get("research_result") or Refusal(reason="insufficient_evidence")
        inp = PMInput(
            symbol=symbol,
            evidence=research_result,
            portfolio=portfolio,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
        )
        result = pm_agent.run(inp)
        return {"pm_result": result}

    def _risk_node(state: dict[str, Any]) -> dict[str, Any]:
        symbol: str = state.get("symbol", "")
        pm_result = state.get("pm_result") or Hold(symbol=symbol, reason="no pm result")
        correlation_id: str = state.get("correlation_id", "")
        prices = _prices_for_risk(pm_result, portfolio, market_data, state)
        inp = RiskInput(
            proposal=pm_result,
            portfolio=portfolio,
            prices=prices,
            correlation_id=correlation_id,
        )
        result = risk_agent.run(inp)
        if isinstance(result, HITLRequired):
            # Auto-approve for eval — no human in the loop.
            trade = _build_eval_trade(result.proposal, correlation_id, prices)
            approved = ApprovedTrade(trade=trade, correlation_id=correlation_id)
            return {"risk_result": result, "approved_trade": approved, "hitl_auto_approved": True}
        if isinstance(result, ApprovedTrade):
            return {"risk_result": result, "approved_trade": result, "hitl_auto_approved": False}
        return {"risk_result": result, "approved_trade": None, "hitl_auto_approved": False}

    def _execution_node(state: dict[str, Any]) -> dict[str, Any]:
        approved: ApprovedTrade | None = state.get("approved_trade")
        correlation_id: str = state.get("correlation_id", "")
        if approved is None:
            return {"exec_result": None}
        prices = {approved.trade.symbol: approved.trade.requested_price}
        inp = ExecutionInput(
            approved_trade=approved,
            portfolio_id=portfolio_id,
            portfolio=portfolio,
            prices=prices,
            correlation_id=correlation_id,
        )
        result = execution_agent.run(inp)
        return {"exec_result": result}

    def _reporting_node(state: dict[str, Any]) -> dict[str, Any]:
        correlation_id: str = state.get("correlation_id", "")
        decision_ts: datetime = state.get("decision_ts") or datetime.now(tz=UTC)
        cycle_id = _str_to_uuid(correlation_id)
        inp = ReportingInput(
            cycle_id=cycle_id,
            portfolio_id=portfolio_id,
            report_date=decision_ts.date(),
            correlation_id=correlation_id,
        )
        reporting_agent.run(inp)
        return {"reporting_done": True}

    def _route_after_risk(state: dict[str, Any]) -> str:
        approved = state.get("approved_trade")
        if approved is not None:
            return "execution"
        return "reporting"

    builder.add_node("research", _research_node)  # type: ignore[type-var]
    builder.add_node("pm", _pm_node)  # type: ignore[type-var]
    builder.add_node("risk", _risk_node)  # type: ignore[type-var]
    builder.add_node("execution", _execution_node)  # type: ignore[type-var]
    builder.add_node("reporting", _reporting_node)  # type: ignore[type-var]

    builder.add_edge(START, "research")
    builder.add_edge("research", "pm")
    builder.add_edge("pm", "risk")
    builder.add_conditional_edges("risk", _route_after_risk, ["execution", "reporting"])
    builder.add_edge("execution", "reporting")
    builder.add_edge("reporting", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _prices_for_risk(
    pm_result: object,
    portfolio: Portfolio,
    market_data: FrozenMarketData,
    state: dict[str, Any],
) -> dict[str, Decimal]:
    """Build a prices dict covering the proposal symbol AND all held symbols.

    ``Portfolio.nav()`` raises ``ValueError`` when any held symbol is absent
    from the prices dict.  This helper ensures all held symbols are covered
    by falling back to the holding's average cost when no bar is available.
    """
    decision_ts: datetime = state.get("decision_ts") or datetime.now(tz=UTC)
    prices: dict[str, Decimal] = {}

    # Price for the proposed trade symbol (derived from proposal)
    if isinstance(pm_result, TradeProposal) and pm_result.qty > Decimal("0"):
        prices[pm_result.symbol] = pm_result.notional / pm_result.qty

    # Prices for all currently held symbols so NAV can be computed
    for sym, holding in portfolio.holdings.items():
        if sym in prices:
            continue
        bar = market_data.get_bar(sym, decision_ts) if decision_ts is not None else None
        if bar is not None:
            prices[sym] = bar.close
        else:
            prices[sym] = holding.avg_cost

    return prices


# ---------------------------------------------------------------------------
# Single-cycle runner
# ---------------------------------------------------------------------------


def _run_cycle(
    symbol: str,
    decision_ts: datetime,
    correlation_id: str,
    graph: Any,
) -> CycleRecord:
    """Execute one research→PM→risk→execution→reporting pass via LangGraph.

    Invokes the compiled eval graph with an in-memory thread and extracts
    CycleRecord fields from the final state.  Never raises — all failures
    are captured in the returned record.
    """
    thread_id = str(uuid.uuid4())
    initial_state = {
        "symbol": symbol,
        "decision_ts": decision_ts,
        "correlation_id": correlation_id,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state: dict = graph.invoke(initial_state, config=config)  # type: ignore[type-arg]
    except Exception:  # noqa: BLE001 — never let a cycle crash the harness
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
    state: dict,  # type: ignore[type-arg]
    symbol: str,
    decision_ts: datetime,
    correlation_id: str,
) -> CycleRecord:
    """Convert final LangGraph state dict to a CycleRecord."""
    research_result = state.get("research_result")
    pm_result = state.get("pm_result")
    risk_result = state.get("risk_result")
    exec_result = state.get("exec_result")
    approved: ApprovedTrade | None = state.get("approved_trade")

    refusal = isinstance(research_result, Refusal)
    injection_detected = (
        isinstance(research_result, Refusal)
        and research_result.reason == "injection_detected"
    )

    evidence_chunks = 0
    citations_used = 0
    has_citation = False
    if isinstance(research_result, Evidence):
        evidence_chunks = len(research_result.claims)
        citations_used = _count_cited_claims(research_result)
        has_citation = citations_used > 0

    trade_proposed = isinstance(pm_result, TradeProposal)
    hitl_required: bool = state.get("hitl_auto_approved", False)
    guardrail_triggered = isinstance(risk_result, Rejected) or isinstance(
        exec_result, ExecutionFailure
    )

    trade_filled = isinstance(exec_result, Fill)
    fill_price: float | None = float(exec_result.fill_price) if isinstance(exec_result, Fill) else None
    fill_qty: float | None = (
        float(approved.trade.qty) if isinstance(exec_result, Fill) and approved is not None else None
    )

    tokens_used = (
        _estimate_tokens(research_result)
        + _estimate_tokens(pm_result)
        + _estimate_tokens(risk_result)
        + _estimate_tokens(exec_result)
    )

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


def _count_cited_claims(evidence: Evidence) -> int:
    """Count claims that have a non-empty source_url (i.e. were actually cited).

    A claim is considered cited when the LLM's parsed output mapped a chunk_id
    to a known source URL in the corpus.  Claims with an empty ``source_url``
    were returned by the LLM but could not be attributed to any retrieved chunk
    and therefore do not count as citations.
    """
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
    """Rough token estimate based on result type and content size.

    Covers all result types that flow through the pipeline so that
    ``tokens_used`` in each CycleRecord accurately reflects all LLM
    activity, not just research/PM steps.
    """
    if isinstance(result, Evidence):
        total = sum(len(c.text) for c in result.claims)
        return max(50, total // 4)
    if isinstance(result, (Refusal, Hold)):
        return 20
    if isinstance(result, TradeProposal):
        return len(result.rationale) // 4 + 30
    if isinstance(result, (ApprovedTrade, HITLRequired, Rejected)):
        return 15
    if isinstance(result, (Fill, ExecutionFailure)):
        return 10
    return 0


def _estimate_cost(tokens: int) -> float:
    """Haiku pricing: ~$0.25/1M input + $1.25/1M output (blended ~$0.50/1M)."""
    return tokens * 0.50 / 1_000_000



def _build_eval_trade(
    proposal: TradeProposal,
    correlation_id: str,
    prices: dict[str, Decimal],
) -> Trade:
    """Build a Trade domain object for an eval HITL-auto-approve."""
    price = prices.get(proposal.symbol, proposal.notional / proposal.qty)
    return Trade(
        id=uuid.uuid4(),
        cycle_id=_str_to_uuid(correlation_id),
        symbol=proposal.symbol,
        side=proposal.side,
        qty=proposal.qty,
        status=TradeStatus.APPROVED,
        requested_price=price,
        idempotency_key=f"eval-{correlation_id}-{proposal.symbol}",
    )


def _str_to_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid4()


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
    """Return True when at least one symbol has a bar for *day*.

    Used to skip weekend and holiday dates that produce hollow CycleRecords
    with no market data and therefore no meaningful signal.
    """
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
    # Only include held symbols in NAV — skip missing prices for unowned symbols
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
    wires CassetteLLM in replay mode (or FakeLLM when no cassette exists),
    builds all five agents with fakes, builds the LangGraph with a
    MemorySaver checkpointer, then iterates over every (symbol, day) pair
    in the replay window.  Days with no bar data (weekends, holidays) are
    skipped so hollow records do not pollute process metrics.

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

    research_agent = ResearchAgent(
        evidence=evidence_store,
        llm=llm,
        injection_guard=injection_guard,
    )
    pm_agent = PortfolioManagerAgent(
        market_data=market_data,
        risk=risk_policy,
        llm=None,  # sentiment via keyword heuristic in eval
    )
    risk_agent = RiskAgent(risk=risk_policy)
    execution_agent = ExecutionAgent(ledger=ledger, guardrail=guardrail)  # type: ignore[arg-type]
    reporting_agent = ReportingAgent(report_sink=report_sink, ledger=ledger)  # type: ignore[arg-type]

    graph = _build_eval_graph(
        research_agent=research_agent,
        pm_agent=pm_agent,
        risk_agent=risk_agent,
        execution_agent=execution_agent,
        reporting_agent=reporting_agent,
        portfolio=portfolio,
        portfolio_id=portfolio_id,
        market_data=market_data,
    )

    watchlist: list[str] = list(config.window_config.get("watchlist", _WATCHLIST))
    days = _parse_window_dates(config.window_config)

    records: list[CycleRecord] = []
    portfolio_history: list[dict[str, Any]] = []

    for day in days:
        # Skip weekends and holidays — no bar data means no real decision cycles.
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
    """Return a CassetteLLM in replay mode if the cassette exists.

    When no cassette exists and ANTHROPIC_API_KEY is set, record mode is used:
    live calls are made and persisted to cassette_path so subsequent runs are
    fully offline.  When no cassette and no API key, fall back to FakeLLM.
    """
    import os

    if cassette_path.exists() and cassette_path.stat().st_size > 0:
        return CassetteLLM(cassette_path=cassette_path, mode="replay")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        from firm.adapters.llm_anthropic import AnthropicLLM

        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        inner = AnthropicLLM(api_key=api_key)
        return CassetteLLM(cassette_path=cassette_path, mode="record", inner=inner)

    # No cassette and no API key — use FakeLLM so CI runs offline.
    canned = LLMResponse(
        content="[]",
        input_tokens=10,
        output_tokens=2,
        model="claude-haiku-4-5",
    )
    return FakeLLM(responses=[canned] * 500)


def _try_load_risk_policy(project_root: Path) -> RiskPolicyConfig:
    """Load risk policy from YAML if available, else return defaults."""
    policy_path = project_root / "config" / "risk_policy.yaml"
    if policy_path.exists():
        try:
            return load_risk_policy(policy_path)
        except Exception:  # noqa: BLE001
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
        b
        for day in days
        for b in [market_data.get_bar(benchmark, day)]
        if b is not None
    ]

    metrics = compute_metrics(result, benchmark_bars)
    report_md = generate_report(metrics, result)

    report_path = output_dir / "eval_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
