"""Shared live-pipeline construction for the web backend.

Extracts the production graph wiring from cli.py so both the CLI ``run``
command and the web ``POST /api/run`` handler can share it without
importing each other.

The key difference between CLI and web mode:
  - CLI: blocks on ``input()`` for HITL decisions.
  - Web: leaves the graph at the interrupt checkpoint and returns immediately;
    the dashboard polls ``GET /api/approvals/pending`` and resumes via
    ``POST /api/approvals/{thread_id}``.

Public API:
  build_live_graph()   — wire adapters + build_graph; returns (graph, portfolio_id).
  pending_approvals()  — inspect the PostgresSaver for active interrupts.
  resume_approval()    — resume an interrupted graph thread via Command.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveGraph:
    """Holds the compiled graph and its wired portfolio_id."""

    graph: Any  # CompiledStateGraph (avoid importing heavy LangGraph at import time)
    portfolio_id: uuid.UUID
    checkpointer: Any  # PostgresSaver
    engine: Any  # SQLAlchemy Engine
    ledger: Any  # LedgerRepository — for pending-run registry and ensure_portfolio


@dataclass(frozen=True)
class InterruptedThread:
    """One LangGraph thread paused on a HITL interrupt."""

    thread_id: str
    correlation_id: str
    symbol: str
    notional: str
    interrupt_payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_STARTING_CASH = Decimal("100000")


def build_live_graph(settings: Any) -> LiveGraph:
    """Wire all live adapters and return a ready-to-use graph + portfolio_id.

    Equivalent to cli._build_live_pipeline but returns a structured object
    so the web layer can access the checkpointer and engine independently.

    Also calls ``ledger.ensure_portfolio()`` so GET /api/portfolio always
    finds a row with the real starting cash rather than returning zeros.
    """
    root = _project_root()
    risk_policy_config = _safe_load_risk_policy(root)

    engine = _build_engine(settings.database_url)
    portfolio_id, guardrail, injection_guard, ledger = _build_live_domain_objects(
        risk_policy_config, engine
    )

    # Idempotently persist the portfolio row so the dashboard shows real NAV.
    ledger.ensure_portfolio(portfolio_id, _STARTING_CASH)

    checkpointer = _build_checkpointer(settings.database_url)
    ports = _build_ports(
        root, risk_policy_config, settings, portfolio_id, guardrail, injection_guard, ledger
    )

    from firm.orchestration.graph import build_graph

    graph = build_graph(checkpointer=checkpointer, ports=ports)
    return LiveGraph(
        graph=graph,
        portfolio_id=portfolio_id,
        checkpointer=checkpointer,
        engine=engine,
        ledger=ledger,
    )


def pending_approvals(live_graph: LiveGraph) -> list[InterruptedThread]:
    """Return threads currently paused on a HITL interrupt.

    Reads the pending_runs registry (written by run_cycle_background) and then
    queries LangGraph state for each registered thread.  Only threads whose
    checkpoint has an active ``interrupts`` list are returned.

    Falls back to the old list_namespaces scan when the registry is empty or
    unavailable (e.g. in tests that pre-date the registry table).

    Returns an empty list when no threads are pending or on any error.
    """
    try:
        return _scan_via_registry(live_graph)
    except Exception:
        return []


def resume_approval(
    live_graph: LiveGraph,
    thread_id: str,
    decision: str,
    edited_qty: Decimal | None,
) -> dict[str, Any]:
    """Resume an interrupted graph thread with the operator decision.

    Builds a LangGraph ``Command`` equivalent to what the CLI does in
    ``_invoke_live_symbol`` after ``_console_hitl_prompt`` returns.  Also
    removes the thread from the pending_runs registry so it no longer appears
    in GET /api/approvals/pending.

    Returns the resolved cycle outcome from the final graph state.
    """
    from firm.orchestration.hitl import HITLDecision, resume_decision

    hitl_decision = _decision_to_hitl_decision(decision)
    final_state = resume_decision(live_graph.graph, thread_id, hitl_decision)

    # Remove from registry now that the thread has been resolved.
    _unregister_pending_run(live_graph, thread_id)

    return {
        "thread_id": thread_id,
        "hitl_status": HITLDecision(hitl_decision).hitl_status,
        "outcome": final_state.get("cycle_outcome", "unknown"),
    }


# ---------------------------------------------------------------------------
# Private helpers — graph construction
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def _build_engine(database_url: str) -> Any:
    from sqlalchemy import create_engine

    from firm.persistence.db_url import to_sqlalchemy_url

    return create_engine(to_sqlalchemy_url(database_url))


def _build_checkpointer(database_url: str) -> Any:
    import psycopg
    import psycopg.rows

    from firm.orchestration.checkpointer import _normalise_database_url, setup_checkpointer

    live_url = _normalise_database_url(database_url)
    pg_conn = psycopg.connect(  # pyright: ignore[reportArgumentType]
        live_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=psycopg.rows.dict_row,
    )
    return setup_checkpointer(pg_conn)


def _build_live_domain_objects(
    risk_policy_config: Any,
    engine: Any,
) -> tuple[uuid.UUID, Any, Any, Any]:
    """Return (portfolio_id, guardrail, injection_guard, ledger).

    Always uses FIRM_PORTFOLIO_ID so the web runtime shares persistent
    portfolio state with the CLI run command across restarts.
    """
    from firm.domain import RiskPolicy
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
    from firm.persistence.ledger import FIRM_PORTFOLIO_ID, LedgerRepository

    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(risk_policy_config.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(risk_policy_config.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(risk_policy_config.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(risk_policy_config.hitl_threshold_pct)),
    )
    guardrail = LedgerGuardrail(domain_policy)
    injection_guard = InjectionGuard()
    ledger = LedgerRepository(engine)
    return FIRM_PORTFOLIO_ID, guardrail, injection_guard, ledger


def _build_ports(
    root: Path,
    risk_policy_config: Any,
    settings: Any,
    portfolio_id: uuid.UUID,
    guardrail: Any,
    injection_guard: Any,
    ledger: Any,
) -> Any:
    """Assemble NodePorts with all live adapters."""
    import psycopg

    from firm.adapters.evidence_pgvector import PgvectorEvidenceStore
    from firm.adapters.llm_anthropic import AnthropicLLM
    from firm.adapters.llm_offline import GracefulLLM
    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.adapters.market_data_live import LiveMarketData
    from firm.adapters.report import ExcelReportSink, MultiReportSink, SlackReportSink
    from firm.domain.guardrails import TokenBudgetCircuitBreaker
    from firm.orchestration.checkpointer import _normalise_database_url
    from firm.orchestration.nodes import NodePorts
    from firm.services.calendar import NYSECalendar

    live_url = _normalise_database_url(settings.database_url)
    evidence_conn = psycopg.connect(live_url)
    evidence_store = PgvectorEvidenceStore(evidence_conn)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_sink = MultiReportSink(
        sinks=[
            ExcelReportSink(output_dir=reports_dir),
            SlackReportSink(channel=getattr(settings, "slack_channel", "#trading-desk")),
        ]
    )

    raw_llm = GracefulLLM(AnthropicLLM(api_key=settings.anthropic_api_key))
    llm = TokenBudgetLLM(
        inner=raw_llm,
        breaker=TokenBudgetCircuitBreaker(),
        budget=risk_policy_config.token_budget_per_cycle,
        report_sink=report_sink,
    )

    from firm.persistence.ledger import FIRM_PORTFOLIO_ID

    portfolio = ledger.get_portfolio(FIRM_PORTFOLIO_ID)
    return NodePorts(
        evidence=evidence_store,
        llm=llm,
        market_data=LiveMarketData(),
        ledger=ledger,
        report_sink=report_sink,
        guardrail=guardrail,
        injection_guard=injection_guard,
        risk_policy=risk_policy_config,
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        calendar=NYSECalendar(),
    )


def _safe_load_risk_policy(root: Path) -> Any:
    from firm.config.settings import RiskPolicyConfig, load_risk_policy

    policy_path = root / "config" / "risk_policy.yaml"
    if policy_path.exists():
        try:
            return load_risk_policy(policy_path)
        except Exception:
            pass
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
# Private helpers — HITL inspection (registry-based)
# ---------------------------------------------------------------------------


def _scan_via_registry(live_graph: LiveGraph) -> list[InterruptedThread]:
    """Enumerate threads from the pending_runs registry and inspect each for interrupts.

    Each registered thread is checked via graph.get_state().  Only those whose
    checkpoint has an active ``interrupts`` list are returned.  Threads that
    have completed (no interrupts) are removed from the registry automatically.
    """
    ledger = live_graph.ledger
    if ledger is None:
        return []

    registered = ledger.list_pending_runs()
    results: list[InterruptedThread] = []
    for thread_id, correlation_id, symbol in registered:
        item = _inspect_thread(live_graph.graph, thread_id, correlation_id, symbol)
        if item is not None:
            results.append(item)
        else:
            # Thread completed without a HITL interrupt — clean up the registry.
            _unregister_pending_run(live_graph, thread_id)
    return results


def _inspect_thread(
    graph: Any,
    thread_id: str,
    correlation_id: str,
    symbol: str,
) -> InterruptedThread | None:
    """Return an InterruptedThread if *thread_id* has a pending interrupt, else None."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    try:
        run_state = graph.get_state(config)
    except Exception:
        return None
    return _extract_interrupt(thread_id, run_state, correlation_id, symbol)


def _extract_interrupt(
    thread_id: str,
    run_state: Any,
    correlation_id: str,
    symbol: str,
) -> InterruptedThread | None:
    """Extract interrupt payload from a graph state if one is present."""
    if not (run_state.next and run_state.tasks):
        return None
    for task in run_state.tasks:
        interrupts = getattr(task, "interrupts", None)
        if interrupts:
            payload: dict[str, Any] = interrupts[0].value if interrupts else {}
            return _build_interrupted_thread(thread_id, run_state, payload, correlation_id, symbol)
    return None


def _build_interrupted_thread(
    thread_id: str,
    run_state: Any,
    payload: dict[str, Any],
    correlation_id: str,
    symbol: str,
) -> InterruptedThread:
    state_values: dict[str, Any] = run_state.values if run_state.values else {}
    effective_correlation = state_values.get("correlation_id") or correlation_id
    effective_symbol = state_values.get("symbol") or symbol
    proposal = payload.get("trade_proposal") or state_values.get("trade_proposal") or {}
    notional = str(proposal.get("notional", "unknown"))
    return InterruptedThread(
        thread_id=thread_id,
        correlation_id=effective_correlation,
        symbol=effective_symbol,
        notional=notional,
        interrupt_payload=payload,
    )


def _unregister_pending_run(live_graph: LiveGraph, thread_id: str) -> None:
    """Remove *thread_id* from the pending-run registry; no-op on failure."""
    ledger = live_graph.ledger
    if ledger is None:
        return
    try:
        ledger.delete_pending_run(thread_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private helpers — resume
# ---------------------------------------------------------------------------


def _decision_to_hitl_decision(decision: str) -> str:
    """Map the web request decision to a structured HITLDecision value.

    ``approve`` executes the recommended action; ``reject`` becomes an explicit
    hold override (no trade) under the always-pause model.  The richer override
    actions (override:buy/sell) are exposed by the resume_decision interface and
    will be surfaced in the dashboard in a follow-up.
    """
    from firm.orchestration.hitl import HITLDecision

    if decision == "approve":
        return HITLDecision.APPROVE.value
    return HITLDecision.OVERRIDE_HOLD.value


# ---------------------------------------------------------------------------
# Background task helper — used by POST /api/run
# ---------------------------------------------------------------------------


def run_cycle_background(
    live_graph: LiveGraph,
    symbol: str,
    decision_ts: str,
    thread_id: str,
    force_buy: bool = False,
) -> None:
    """Run one graph cycle in the background (non-blocking HITL mode).

    Unlike the CLI path, we do NOT block on ``input()``.  When a HITL interrupt
    fires, the graph stops at the checkpoint.  The web operator sees it in
    ``GET /api/approvals/pending`` and resumes via ``POST /api/approvals/{thread_id}``.

    When *force_buy* is True, a synthetic high-conviction BUY research_plan is
    injected so the pipeline always proposes a trade above the HITL threshold.
    This is a named demo/override path — it does NOT change default behaviour.

    Args:
        live_graph: Wired graph + engine + ledger.
        symbol: Ticker to analyse.
        decision_ts: ISO-8601 decision timestamp string.
        thread_id: LangGraph checkpoint thread ID.
        force_buy: If True, inject a synthetic BUY plan so HITL fires reliably.
    """
    import uuid as _uuid

    from firm.observability.tracing import reset_correlation_id, set_correlation_id

    correlation_id = str(_uuid.uuid4())
    token = set_correlation_id(correlation_id)
    try:
        initial_state: dict[str, Any] = _build_initial_state(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
            force_buy=force_buy,
        )
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        # Register before streaming so the thread is visible immediately.
        _register_pending_run(live_graph, thread_id, correlation_id, symbol)
        # Stream until the graph either completes or hits a HITL interrupt.
        # We do NOT resume here — the web operator handles that.
        for _event in live_graph.graph.stream(initial_state, config=config, stream_mode="values"):
            pass
    except Exception:
        pass
    finally:
        reset_correlation_id(token)


def _build_initial_state(
    symbol: str,
    decision_ts: str,
    correlation_id: str,
    force_buy: bool,
) -> dict[str, Any]:
    """Construct the initial GraphState dict.

    When *force_buy* is True, sets ``force_buy=True`` in state so the
    research_manager_node short-circuits to a synthetic BUY plan (skipping
    the LLM call) and the pm_node sizes a position large enough to trigger HITL.
    """
    return {
        "symbol": symbol,
        "decision_ts": decision_ts,
        "correlation_id": correlation_id,
        "trigger_type": "scheduled",
        "force_buy": force_buy,
    }


def _register_pending_run(
    live_graph: LiveGraph,
    thread_id: str,
    correlation_id: str,
    symbol: str,
) -> None:
    """Write a pending_run row; silently swallow errors (non-critical path)."""
    ledger = live_graph.ledger
    if ledger is None:
        return
    try:
        ledger.register_pending_run(
            thread_id=thread_id,
            correlation_id=correlation_id,
            symbol=symbol,
        )
    except Exception:
        pass
