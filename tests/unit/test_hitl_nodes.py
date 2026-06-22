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

from firm.config.settings import RiskPolicyConfig
from firm.orchestration.graph import _route_after_risk, build_graph
from firm.orchestration.nodes import _hitl_exceeds_threshold, make_risk_node
from firm.orchestration.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    risk_fn = make_risk_node(policy)
    result = risk_fn(state, RunnableConfig())

    assert result.get("cycle_outcome") == "error"
    assert "trade_proposal missing" in (result.get("error") or "")


def test_risk_node_below_threshold_sets_approved_trade() -> None:
    """notional < threshold must set approved_trade and not interrupt."""
    # pm_node returns notional=1000, which exceeds threshold=500.
    # Override: use a policy with a very high threshold so no interrupt fires.
    high_threshold_policy = _make_risk_policy(hitl_threshold_pct=0.20)
    saver2 = MemorySaver()
    graph2 = build_graph(saver2, risk_policy=high_threshold_policy)

    # pm_node produces notional=1000; threshold at 20% of 10000 = 2000 → no interrupt.
    events = list(
        graph2.stream(
            _make_state("1000"),
            _thread_config(str(uuid.uuid4())),
            stream_mode="updates",
        )
    )
    interrupt_events = [e for e in events if "__interrupt__" in e]
    assert not interrupt_events, "notional below threshold must not interrupt"

    final_values = {}
    for event in events:
        for _node, patch in event.items():
            if isinstance(patch, dict):
                final_values.update(patch)
    assert final_values.get("cycle_outcome") == "filled"


def test_risk_node_at_threshold_does_not_interrupt() -> None:
    """notional == threshold must not trigger HITL (strictly-above boundary)."""
    from langchain_core.runnables import RunnableConfig

    policy = _make_risk_policy(hitl_threshold_pct=0.10)  # threshold = 1000
    risk_fn = make_risk_node(policy)

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
    """notional > threshold must emit an __interrupt__ event."""
    policy = _make_risk_policy()  # threshold = 500
    saver = MemorySaver()
    graph = build_graph(saver, risk_policy=policy)

    # pm_node always produces notional=1000 > 500
    events = list(
        graph.stream(
            _make_state("1000"),
            _thread_config(str(uuid.uuid4())),
            stream_mode="updates",
        )
    )
    interrupt_events = [e for e in events if "__interrupt__" in e]
    assert interrupt_events, "notional above threshold must trigger HITL interrupt"
    assert interrupt_events[-1]["__interrupt__"][0].value["type"] == "hitl_request"
