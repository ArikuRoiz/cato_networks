"""Integration tests for durable HITL interrupt/resume via Postgres checkpointer.

These tests satisfy the two mandatory FIRM-5 requirements:
  - ``test_hitl_resumes_after_restart``: the graph checkpoints state before
    the interrupt; a brand-new process (new graph + new checkpointer + same
    Postgres) can resume and reach ``cycle_outcome="filled"``.
  - ``test_hitl_timeout_fails_safe``: an expired approval produces
    ``cycle_outcome="rejected_timeout"`` (never auto-approved).

Both tests use a real ephemeral Postgres instance via testcontainers.

Resume pattern:
    LangGraph's ``interrupt()`` model re-executes the interrupted node from
    the top after ``Command(resume=value, update=state_patches)``.  Callers
    inject ``hitl_status`` via the ``update`` field so the node can read it
    from state on re-entry.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from testcontainers.postgres import PostgresContainer

from firm.adapters.fakes import FakeEvidenceStore, FakeLLM, FakeMarketData, FakeReportSink
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio, RiskPolicy
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.checkpointer import open_connection, setup_checkpointer
from firm.orchestration.nodes import NodePorts
from firm.orchestration.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POSTGRES_IMAGE = "pgvector/pgvector:pg16"

_RISK_POLICY = RiskPolicyConfig(
    max_trade_notional_pct=0.10,
    max_name_concentration_pct=0.25,
    daily_loss_halt_pct=0.03,
    hitl_threshold_pct=0.05,  # threshold = 5% of 10k = 500
    buy_threshold=0.15,
    sell_threshold=-0.10,
    momentum_weight=0.60,
    sentiment_weight=0.40,
    momentum_lookback_days=5,
    max_events_per_symbol_per_hour=3,
    event_relevance_threshold=0.70,
    slippage_bps=5,
    commission_per_share=0.005,
    token_budget_per_cycle=50000,
)


def _make_ports() -> NodePorts:
    """Return NodePorts with all-fake dependencies for HITL integration tests."""
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal("0.10"),
        max_name_concentration_pct=Decimal("0.25"),
        daily_loss_halt_pct=Decimal("0.03"),
        hitl_threshold_pct=Decimal("0.05"),
    )
    return NodePorts(
        evidence=FakeEvidenceStore(),
        llm=FakeLLM(),
        market_data=FakeMarketData(),
        ledger=None,  # type: ignore[arg-type]
        report_sink=FakeReportSink(),
        guardrail=LedgerGuardrail(domain_policy),
        injection_guard=InjectionGuard(),
        risk_policy=_RISK_POLICY,
        portfolio_id=uuid.uuid4(),
        portfolio=Portfolio(cash=Decimal("10000")),
    )


def _make_initial_state(correlation_id: str) -> GraphState:
    """Return a ``GraphState`` with trade_proposal=notional=1000 that triggers HITL.

    hitl_threshold_pct=0.05 × NAV(10000) = 500, so notional=1000 > 500 guarantees
    the interrupt path when the risk node runs via the minimal inject→risk graph.
    """
    return GraphState(
        correlation_id=correlation_id,
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal={
            "symbol": "NVDA",
            "side": "buy",
            "qty": "10",
            "notional": "1000",  # > threshold=500
            "rationale": "stub",
        },
        approved_trade=None,
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )


def _build_minimal_graph(checkpointer: object) -> object:
    """Build a minimal inject→risk→(execution|done) graph for HITL checkpoint tests.

    The full production graph requires a LLM cassette, market data, and many
    node dependencies.  For HITL durability tests we only need risk_node to
    fire interrupt() and resume from the Postgres checkpoint, then route to
    an execution stub (approved) or end (rejected/timeout).
    """
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph

    from firm.orchestration.nodes import make_risk_node

    risk_fn = make_risk_node(_make_ports())

    def _inject(state: GraphState) -> dict:  # type: ignore[type-arg]
        return {}

    def _execution_stub(state: GraphState) -> dict:  # type: ignore[type-arg]
        """Minimal execution: mark the cycle filled without real ledger IO."""
        return {"cycle_outcome": "filled"}

    def _route_after_risk(state: GraphState) -> str:
        if state.get("approved_trade") is not None:
            return "execution"
        return END  # type: ignore[return-value]

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]
    builder.add_node("inject", _inject)
    builder.add_node("risk", risk_fn)
    builder.add_node("execution", _execution_stub)
    builder.add_edge(START, "inject")
    builder.add_edge("inject", "risk")
    builder.add_conditional_edges("risk", _route_after_risk, ["execution", END])
    builder.add_edge("execution", END)
    return builder.compile(checkpointer=checkpointer)


def _thread_config(thread_id: str) -> dict:  # type: ignore[type-arg]
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_hitl_resumes_after_restart() -> None:
    """Graph checkpoints state before the interrupt; a new process resumes to
    ``cycle_outcome='filled'`` after an approved HITL decision.

    Steps:
      1. Start a run whose notional triggers HITL.
      2. Assert the graph stopped at the risk node (interrupt).
      3. Simulate a process restart: create NEW graph + NEW checkpointer
         connected to the same Postgres.
      4. Resume with ``hitl_status='approved'`` via ``Command``.
      5. Assert ``cycle_outcome='filled'``.
    """
    from langgraph.types import Command

    correlation_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    with PostgresContainer(_POSTGRES_IMAGE) as pg:
        # Pass the raw testcontainers URL directly — open_connection calls
        # _normalise_database_url internally and handles any SQLAlchemy dialect
        # prefix regardless of testcontainers' URL format.
        db_url = pg.get_connection_url()

        initial_state = _make_initial_state(correlation_id)
        config = _thread_config(thread_id)

        # --- First "process": start the graph run ---
        with open_connection(db_url) as conn_1:
            checkpointer_1 = setup_checkpointer(conn_1)
            graph_1 = _build_minimal_graph(checkpointer_1)

            # The graph should interrupt at risk_node — collect all events
            events = list(graph_1.stream(initial_state, config, stream_mode="updates"))

            # Verify the graph interrupted (last event should be an interrupt signal)
            interrupt_events = [e for e in events if "__interrupt__" in e]
            assert interrupt_events, (
                "Expected the graph to emit an __interrupt__ event from risk_node; "
                f"got events: {events}"
            )

            interrupt_payload = interrupt_events[-1]["__interrupt__"]
            assert len(interrupt_payload) > 0
            assert interrupt_payload[0].value["type"] == "hitl_request"

            # Verify the graph really paused — no cycle_outcome yet
            snapshot = graph_1.get_state(config)
            assert snapshot.values.get("cycle_outcome") is None, (
                "cycle_outcome must be None while HITL is pending"
            )

        # --- Simulate process restart: new graph + new checkpointer, same DB ---
        with open_connection(db_url) as conn_2:
            checkpointer_2 = setup_checkpointer(conn_2)
            graph_2 = _build_minimal_graph(checkpointer_2)

            # Resume with approved status injected into state
            resume_cmd = Command(
                resume={"decision": "approved"},
                update={"hitl_status": "approved"},
            )
            list(graph_2.stream(resume_cmd, config, stream_mode="updates"))

            final_state = graph_2.get_state(config)
            assert final_state.values.get("cycle_outcome") == "filled", (
                f"Expected cycle_outcome='filled' after approved HITL; "
                f"got {final_state.values.get('cycle_outcome')!r}"
            )


@pytest.mark.integration
def test_hitl_timeout_fails_safe() -> None:
    """An expired HITL decision must produce ``cycle_outcome='rejected_timeout'``.

    Timeout must never result in auto-approval (fail-safe, not fail-open).

    Steps:
      1. Start a run whose notional triggers HITL.
      2. Assert the graph stopped at the risk node (interrupt).
      3. Simulate a process restart: create NEW graph + NEW checkpointer
         connected to the same Postgres (same as ``test_hitl_resumes_after_restart``
         but with ``hitl_status='expired'``).
      4. Resume with ``hitl_status='expired'`` via ``Command``.
      5. Assert ``cycle_outcome='rejected_timeout'`` (never auto-approved).
    """
    from langgraph.types import Command

    correlation_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    with PostgresContainer(_POSTGRES_IMAGE) as pg:
        # Pass the raw testcontainers URL directly — open_connection calls
        # _normalise_database_url internally and handles any SQLAlchemy dialect
        # prefix regardless of testcontainers' URL format.
        db_url = pg.get_connection_url()

        initial_state = _make_initial_state(correlation_id)
        config = _thread_config(thread_id)

        # --- First "process": start the graph run ---
        with open_connection(db_url) as conn_1:
            checkpointer_1 = setup_checkpointer(conn_1)
            graph_1 = _build_minimal_graph(checkpointer_1)

            # Start and interrupt
            events = list(graph_1.stream(initial_state, config, stream_mode="updates"))
            interrupt_events = [e for e in events if "__interrupt__" in e]
            assert interrupt_events, "Expected interrupt from risk_node"

        # --- Simulate process restart: new graph + new checkpointer, same DB ---
        with open_connection(db_url) as conn_2:
            checkpointer_2 = setup_checkpointer(conn_2)
            graph_2 = _build_minimal_graph(checkpointer_2)

            # Resume with expired status — must reject, not approve
            resume_cmd = Command(
                resume={"decision": "expired"},
                update={"hitl_status": "expired"},
            )
            list(graph_2.stream(resume_cmd, config, stream_mode="updates"))

            final_state = graph_2.get_state(config)
            outcome = final_state.values.get("cycle_outcome")
            assert outcome == "rejected_timeout", (
                f"Expired HITL must produce 'rejected_timeout', not {outcome!r}. "
                "Timeout must never auto-approve (fail-safe)."
            )
