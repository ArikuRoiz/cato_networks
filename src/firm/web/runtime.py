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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

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


def build_live_graph(settings: Any) -> LiveGraph:
    """Wire all live adapters and return a ready-to-use graph + portfolio_id.

    Equivalent to cli._build_live_pipeline but returns a structured object
    so the web layer can access the checkpointer and engine independently.
    """
    root = _project_root()
    risk_policy_config = _safe_load_risk_policy(root)

    engine = _build_engine(settings.database_url)
    portfolio_id, guardrail, injection_guard, ledger = _build_live_domain_objects(
        risk_policy_config, engine
    )
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
    )


def pending_approvals(live_graph: LiveGraph) -> list[InterruptedThread]:
    """Scan the PostgresSaver for threads paused on a HITL interrupt.

    LangGraph stores checkpoint state per thread.  We iterate the stored
    checkpoints, check whether the thread has pending interrupts, and surface
    the relevant proposal metadata from the interrupt payload.

    Returns an empty list when no threads are pending or on any error.
    """
    try:
        return _scan_for_interrupts(live_graph)
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
    ``_invoke_live_symbol`` after ``_console_hitl_prompt`` returns.

    Returns the resolved cycle outcome from the final graph state.
    """
    from langgraph.types import Command

    hitl_status = _decision_to_hitl_status(decision, edited_qty)
    resume_value = _build_resume_value(decision, edited_qty)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    resume_cmd: Any = Command(
        resume=resume_value,
        update={"hitl_status": hitl_status},
    )

    final_state: dict[str, Any] = {}
    for event in live_graph.graph.stream(resume_cmd, config=config, stream_mode="values"):
        final_state = event

    return {
        "thread_id": thread_id,
        "hitl_status": hitl_status,
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
    """Return (portfolio_id, guardrail, injection_guard, ledger)."""
    from firm.domain import RiskPolicy
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
    from firm.persistence.ledger import LedgerRepository

    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(risk_policy_config.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(risk_policy_config.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(risk_policy_config.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(risk_policy_config.hitl_threshold_pct)),
    )
    guardrail = LedgerGuardrail(domain_policy)
    injection_guard = InjectionGuard()
    ledger = LedgerRepository(engine)
    return uuid.uuid4(), guardrail, injection_guard, ledger


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
    from firm.domain import Portfolio
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

    portfolio = Portfolio(cash=Decimal("100000"))
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
# Private helpers — HITL inspection
# ---------------------------------------------------------------------------


def _scan_for_interrupts(live_graph: LiveGraph) -> list[InterruptedThread]:
    """Inspect the checkpointer for threads with active interrupts."""
    results: list[InterruptedThread] = []
    try:
        # PostgresSaver exposes list_namespaces / list; we iterate stored checkpoints.
        checkpointer: PostgresSaver = live_graph.checkpointer
        for ns in checkpointer.list_namespaces():
            item = _inspect_namespace(live_graph.graph, ns)
            if item is not None:
                results.append(item)
    except Exception:
        # Graceful degradation: if the checkpointer API differs, return empty.
        pass
    return results


def _inspect_namespace(graph: Any, ns: tuple[str, ...]) -> InterruptedThread | None:
    """Return an InterruptedThread if *ns* has a pending interrupt, else None."""
    if not ns:
        return None
    thread_id = ns[0]
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    try:
        run_state = graph.get_state(config)
    except Exception:
        return None
    return _extract_interrupt(thread_id, run_state)


def _extract_interrupt(thread_id: str, run_state: Any) -> InterruptedThread | None:
    """Extract interrupt payload from a graph state if one is present."""
    if not (run_state.next and run_state.tasks):
        return None
    for task in run_state.tasks:
        interrupts = getattr(task, "interrupts", None)
        if interrupts:
            payload: dict[str, Any] = interrupts[0].value if interrupts else {}
            return _build_interrupted_thread(thread_id, run_state, payload)
    return None


def _build_interrupted_thread(
    thread_id: str,
    run_state: Any,
    payload: dict[str, Any],
) -> InterruptedThread:
    state_values: dict[str, Any] = run_state.values if run_state.values else {}
    correlation_id = state_values.get("correlation_id", thread_id)
    symbol = state_values.get("symbol", "unknown")
    proposal = payload.get("trade_proposal") or state_values.get("trade_proposal") or {}
    notional = str(proposal.get("notional", "unknown"))
    return InterruptedThread(
        thread_id=thread_id,
        correlation_id=correlation_id,
        symbol=symbol,
        notional=notional,
        interrupt_payload=payload,
    )


# ---------------------------------------------------------------------------
# Private helpers — resume
# ---------------------------------------------------------------------------


def _decision_to_hitl_status(decision: str, edited_qty: Decimal | None) -> str:
    from firm.domain.enums import HITLStatus

    if decision == "approve":
        return HITLStatus.APPROVED
    return HITLStatus.REJECTED


def _build_resume_value(decision: str, edited_qty: Decimal | None) -> str:
    if decision == "approve" and edited_qty is not None:
        return f"edit:{edited_qty}"
    if decision == "approve":
        return "approved"
    return "rejected"


# ---------------------------------------------------------------------------
# Background task helper — used by POST /api/run
# ---------------------------------------------------------------------------


def run_cycle_background(
    live_graph: LiveGraph,
    symbol: str,
    decision_ts: str,
    thread_id: str,
) -> None:
    """Run one graph cycle in the background (non-blocking HITL mode).

    Unlike the CLI path, we do NOT block on ``input()``.  When a HITL interrupt
    fires, the graph stops at the checkpoint.  The web operator sees it in
    ``GET /api/approvals/pending`` and resumes via ``POST /api/approvals/{thread_id}``.
    """
    import uuid as _uuid

    from firm.observability.tracing import reset_correlation_id, set_correlation_id

    correlation_id = str(_uuid.uuid4())
    token = set_correlation_id(correlation_id)
    try:
        initial_state: dict[str, Any] = {
            "symbol": symbol,
            "decision_ts": decision_ts,
            "correlation_id": correlation_id,
        }
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        # Stream until the graph either completes or hits a HITL interrupt.
        # We do NOT resume here — the web operator handles that.
        for _event in live_graph.graph.stream(initial_state, config=config, stream_mode="values"):
            pass
    except Exception:
        pass
    finally:
        reset_correlation_id(token)
