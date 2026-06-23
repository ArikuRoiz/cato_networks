"""Unit tests for HITL orchestration logic in nodes.py and graph.py.

All tests use ``MemorySaver`` — no Postgres required.

Coverage:
  - ``_hitl_exceeds_threshold``: below, at, and above the threshold boundary.
  - ``make_risk_node`` via ``build_graph``:
      * ``trade_proposal=None`` error path.
      * notional below threshold → approved_trade set, no interrupt.
      * notional at threshold (equality) → approved_trade set, no interrupt
        (strictly-above boundary is documented in _hitl_exceeds_threshold).
      * notional above threshold → graph interrupts at risk node.
  - ``_route_after_risk``: routes to "execution" when approved_trade is set,
    routes to "reporting" otherwise.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from langgraph.checkpoint.memory import MemorySaver

from firm.adapters.fakes import FakeEvidenceStore, FakeLLM, FakeMarketData, FakeReportSink
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio, RiskPolicy
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.graph import _route_after_risk, build_graph
from firm.orchestration.nodes import NodePorts, _hitl_exceeds_threshold, make_risk_node
from firm.orchestration.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ports(risk_policy: RiskPolicyConfig) -> NodePorts:
    """Build a NodePorts with all-fake dependencies for unit tests."""
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(risk_policy.max_trade_notional_pct)),
        max_name_concentration_pct=Decimal(str(risk_policy.max_name_concentration_pct)),
        daily_loss_halt_pct=Decimal(str(risk_policy.daily_loss_halt_pct)),
        hitl_threshold_pct=Decimal(str(risk_policy.hitl_threshold_pct)),
    )
    return NodePorts(
        evidence=FakeEvidenceStore(),
        llm=FakeLLM(),
        market_data=FakeMarketData(),
        ledger=None,  # type: ignore[arg-type]
        report_sink=FakeReportSink(),
        guardrail=LedgerGuardrail(domain_policy),
        injection_guard=InjectionGuard(),
        risk_policy=risk_policy,
        portfolio_id=uuid.uuid4(),
        portfolio=Portfolio(cash=Decimal("10000")),
    )


_NAV_ESTIMATE = Decimal("10000")
_HITL_THRESHOLD_PCT = Decimal("0.05")
_THRESHOLD = _HITL_THRESHOLD_PCT * _NAV_ESTIMATE  # == 500


def _make_risk_policy(hitl_threshold_pct: float = 0.05) -> RiskPolicyConfig:
    return RiskPolicyConfig(
        max_trade_notional_pct=0.10,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=hitl_threshold_pct,
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


def _make_state(notional: str) -> GraphState:
    return GraphState(
        correlation_id=str(uuid.uuid4()),
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal={
            "symbol": "NVDA",
            "side": "buy",
            "qty": "10",
            "notional": notional,
            "rationale": "stub",
        },
        approved_trade=None,
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )


def _thread_config(thread_id: str) -> dict:  # type: ignore[type-arg]
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# _hitl_exceeds_threshold — boundary tests
# ---------------------------------------------------------------------------


def test_hitl_threshold_below_does_not_trigger() -> None:
    assert _hitl_exceeds_threshold(Decimal("499"), _THRESHOLD) is False


def test_hitl_threshold_equal_does_not_trigger() -> None:
    # Boundary is strictly above (>), so equality must not trigger HITL.
    assert _hitl_exceeds_threshold(Decimal("500"), _THRESHOLD) is False


def test_hitl_threshold_above_triggers() -> None:
    assert _hitl_exceeds_threshold(Decimal("501"), _THRESHOLD) is True


def test_hitl_threshold_well_above_triggers() -> None:
    assert _hitl_exceeds_threshold(Decimal("1000"), _THRESHOLD) is True


# ---------------------------------------------------------------------------
# _route_after_risk — routing predicate
# ---------------------------------------------------------------------------


def test_route_after_risk_goes_to_execution_when_approved_trade_set() -> None:
    state = GraphState(
        correlation_id="x",
        trigger_type="scheduled",
        symbol="AAPL",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal=None,
        approved_trade={
            "symbol": "AAPL",
            "side": "buy",
            "qty": "5",
            "notional": "100",
            "rationale": "stub",
        },
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )
    assert _route_after_risk(state) == "execution"


def test_route_after_risk_goes_to_reporting_when_no_approved_trade() -> None:
    state = GraphState(
        correlation_id="x",
        trigger_type="scheduled",
        symbol="AAPL",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal=None,
        approved_trade=None,
        hitl_status=None,
        cycle_outcome="rejected",
        error=None,
        token_count=0,
    )
    assert _route_after_risk(state) == "reporting"


# ---------------------------------------------------------------------------
# risk_node (via build_graph + MemorySaver) — no-interrupt paths
# ---------------------------------------------------------------------------


def test_risk_node_missing_trade_proposal_returns_error() -> None:
    """trade_proposal=None must return cycle_outcome='error', not raise."""
    policy = _make_risk_policy()

    state = GraphState(
        correlation_id=str(uuid.uuid4()),
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal=None,
        approved_trade=None,
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )
    # Patch pm_node's output by injecting state with no proposal.
    # We invoke directly against the compiled graph — the pm stub will
    # overwrite trade_proposal, so we need to run only the risk node.
    # Use make_risk_node directly for this path.
    from langchain_core.runnables import RunnableConfig

    risk_fn = make_risk_node(_make_ports(policy))
    result = risk_fn(state, RunnableConfig())

    assert result.get("cycle_outcome") == "error"
    assert "trade_proposal missing" in (result.get("error") or "")


def test_risk_node_below_threshold_sets_approved_trade() -> None:
    """notional < threshold must set approved_trade and not interrupt."""
    # Use make_risk_node directly to avoid the full pipeline; the graph-level
    # interrupt mechanism is tested separately in test_risk_node_above_threshold_interrupts.
    from langchain_core.runnables import RunnableConfig

    high_threshold_policy = _make_risk_policy(hitl_threshold_pct=0.20)  # threshold=2000
    risk_fn = make_risk_node(_make_ports(high_threshold_policy))

    # notional=1000 < threshold=2000 → no interrupt, approved_trade is set.
    state = _make_state("1000")
    result = risk_fn(state, RunnableConfig())

    assert result.get("approved_trade") is not None, "below-threshold trade must be approved"
    assert "cycle_outcome" not in result or result.get("cycle_outcome") is None


def test_risk_node_at_threshold_does_not_interrupt() -> None:
    """notional == threshold must not trigger HITL (strictly-above boundary)."""
    from langchain_core.runnables import RunnableConfig

    policy = _make_risk_policy(hitl_threshold_pct=0.10)  # threshold = 1000
    risk_fn = make_risk_node(_make_ports(policy))

    state: GraphState = GraphState(
        correlation_id=str(uuid.uuid4()),
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        trade_proposal={
            "symbol": "NVDA",
            "side": "buy",
            "qty": "10",
            "notional": "1000",
            "rationale": "stub",
        },
        approved_trade=None,
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )
    result = risk_fn(state, RunnableConfig())
    assert result.get("approved_trade") is not None
    assert "cycle_outcome" not in result or result.get("cycle_outcome") is None


def test_risk_node_above_threshold_interrupts() -> None:
    """notional > threshold must emit an __interrupt__ event.

    Uses a minimal two-node graph (inject → risk) where the inject node copies
    the pre-seeded trade_proposal into state, allowing risk_node to receive it
    without running the full pipeline.
    """
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph

    policy = _make_risk_policy()  # threshold = 500
    ports = _make_ports(policy)
    risk_fn = make_risk_node(ports)

    def _inject_proposal(state: GraphState) -> dict:  # type: ignore[type-arg]
        """Pass the pre-seeded trade_proposal through unchanged."""
        return {}

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]
    builder.add_node("inject", _inject_proposal)
    builder.add_node("risk", risk_fn)
    builder.add_edge(START, "inject")
    builder.add_edge("inject", "risk")
    builder.add_edge("risk", END)
    saver = MemorySaver()
    mini_graph = builder.compile(checkpointer=saver)

    # Pre-seed state with notional=1000 > threshold=500
    state = _make_state("1000")
    config = _thread_config(str(uuid.uuid4()))

    events = list(mini_graph.stream(state, config, stream_mode="updates"))
    interrupt_events = [e for e in events if "__interrupt__" in e]
    assert interrupt_events, "notional above threshold must trigger HITL interrupt"
    assert interrupt_events[-1]["__interrupt__"][0].value["type"] == "hitl_request"
