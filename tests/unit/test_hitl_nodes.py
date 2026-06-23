"""Unit tests for HITL orchestration logic in nodes.py and graph.py.

All tests use ``MemorySaver`` — no Postgres required.

HITL model (every cycle pauses unless ``hitl_mode='threshold'``):
  - default ``hitl_mode='always'`` → ``make_risk_node`` interrupts on every
    recommendation, even below the notional threshold.
  - ``hitl_mode='threshold'`` (legacy) → only proposals whose notional exceeds
    the HITL threshold pause; smaller trades auto-approve.

Coverage:
  - ``_should_pause``: always vs. threshold gating.
  - ``make_risk_node``:
      * ``trade_proposal=None`` error path.
      * threshold mode, notional below threshold → approved_trade set, no interrupt.
      * threshold mode, notional at threshold (equality) → approved_trade set, no interrupt.
      * always mode, notional below threshold → interrupt anyway.
      * notional above threshold → graph interrupts at risk node.
  - ``_route_after_risk``: routes to "execution" when approved_trade is set,
    routes to "reporting" otherwise.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from langgraph.checkpoint.memory import MemorySaver

from firm.adapters.fakes import (
    FakeCalendar,
    FakeEvidenceStore,
    FakeLLM,
    FakeMarketData,
    FakeReportSink,
)
from firm.agents.risk import HITLRequired as AgentHITLRequired
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio, RiskPolicy
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.graph import _route_after_risk
from firm.orchestration.nodes import NodePorts, _should_pause, make_risk_node
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
        calendar=FakeCalendar(is_open=True),
    )


def _make_risk_policy(
    hitl_threshold_pct: float = 0.05,
    hitl_mode: str = "always",
) -> RiskPolicyConfig:
    return RiskPolicyConfig(
        max_trade_notional_pct=0.10,
        max_name_concentration_pct=0.25,
        daily_loss_halt_pct=0.03,
        hitl_threshold_pct=hitl_threshold_pct,
        hitl_mode=hitl_mode,  # type: ignore[arg-type]
        buy_threshold=0.15,
        sell_threshold=-0.10,
        momentum_weight=0.60,
        sentiment_weight=0.40,
        momentum_lookback_days=5,
        max_events_per_symbol_per_hour=3,
        event_relevance_threshold=0.70,
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
# _should_pause — always vs. threshold gating
# ---------------------------------------------------------------------------


class _FakeApproved:
    """Stand-in for an ApprovedTrade-shaped risk result (below threshold)."""


def _hitl_required() -> AgentHITLRequired:
    """A HITLRequired result (notional above the HITL threshold)."""
    from firm.agents.portfolio_manager.schemas import TradeProposal
    from firm.domain.enums import TradeSide

    proposal = TradeProposal(
        symbol="NVDA",
        side=TradeSide.BUY,
        qty=Decimal("10"),
        notional=Decimal("1000"),
        rationale="stub",
    )
    return AgentHITLRequired(proposal=proposal, reason="above threshold", correlation_id="cid")


def test_should_pause_always_mode_pauses_below_threshold() -> None:
    # always mode pauses on every recommendation, even an approved (below-threshold) one.
    assert _should_pause("always", _FakeApproved()) is True


def test_should_pause_threshold_mode_skips_below_threshold() -> None:
    # threshold mode only pauses on HITLRequired; an approved result auto-approves.
    assert _should_pause("threshold", _FakeApproved()) is False


def test_should_pause_threshold_mode_pauses_above_threshold() -> None:
    assert _should_pause("threshold", _hitl_required()) is True


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
    """threshold mode: notional < threshold must set approved_trade and not interrupt."""
    # Use make_risk_node directly to avoid the full pipeline; the graph-level
    # interrupt mechanism is tested separately in test_risk_node_above_threshold_interrupts.
    from langchain_core.runnables import RunnableConfig

    # threshold mode so the below-threshold trade auto-approves (always mode would interrupt).
    high_threshold_policy = _make_risk_policy(
        hitl_threshold_pct=0.20, hitl_mode="threshold"
    )  # threshold=2000
    risk_fn = make_risk_node(_make_ports(high_threshold_policy))

    # notional=1000 < threshold=2000 → no interrupt, approved_trade is set.
    state = _make_state("1000")
    result = risk_fn(state, RunnableConfig())

    assert result.get("approved_trade") is not None, "below-threshold trade must be approved"
    assert "cycle_outcome" not in result or result.get("cycle_outcome") is None


def test_risk_node_always_mode_interrupts_below_threshold() -> None:
    """always mode: even a below-threshold trade must pause for the human.

    interrupt() only works inside a running graph, so drive the node through a
    minimal inject→risk graph and assert an __interrupt__ event is emitted even
    though the notional (1000) is below the HITL threshold (2000).
    """
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph

    policy = _make_risk_policy(hitl_threshold_pct=0.20, hitl_mode="always")  # threshold=2000
    risk_fn = make_risk_node(_make_ports(policy))

    def _inject(state: GraphState) -> dict:  # type: ignore[type-arg]
        return {}

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]
    builder.add_node("inject", _inject)
    builder.add_node("risk", risk_fn)
    builder.add_edge(START, "inject")
    builder.add_edge("inject", "risk")
    builder.add_edge("risk", END)
    mini_graph = builder.compile(checkpointer=MemorySaver())

    state = _make_state("1000")  # below threshold, but always mode pauses anyway
    config = _thread_config(str(uuid.uuid4()))
    events = list(mini_graph.stream(state, config, stream_mode="updates"))
    assert [e for e in events if "__interrupt__" in e], (
        "always mode must interrupt even below the HITL threshold"
    )


def test_risk_node_at_threshold_does_not_interrupt() -> None:
    """threshold mode: notional == threshold must not trigger HITL (strictly-above boundary)."""
    from langchain_core.runnables import RunnableConfig

    policy = _make_risk_policy(hitl_threshold_pct=0.10, hitl_mode="threshold")  # threshold = 1000
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


# ---------------------------------------------------------------------------
# Closed-market gate — execution node must not produce a fill (R5 §1)
# ---------------------------------------------------------------------------


def test_execution_node_closed_market_produces_no_fill() -> None:
    """A cycle with an approved trade but a closed-market decision_ts yields no fill.

    The execution node must gate on NYSE market hours: when the decision_ts
    falls outside regular trading hours the node must return
    ``cycle_outcome='rejected_market_closed'`` without touching the ledger.

    Requirement: R5 §1 — NYSE market-hours gating.
    """
    from firm.domain.enums import CycleOutcome
    from firm.orchestration.nodes import make_execution_node

    policy = _make_risk_policy()
    # closed_market=True means is_open=False → every is_market_open() call returns False
    ports_closed = _make_ports(policy)
    # Replace the always-open calendar with a closed one
    import dataclasses

    ports_closed = dataclasses.replace(ports_closed, calendar=FakeCalendar(is_open=False))

    execution_fn = make_execution_node(ports_closed)

    # Build a state with a valid approved_trade and a weekend decision_ts.
    # 2024-10-26 is a Saturday — closed even without the fake.
    closed_state = GraphState(
        correlation_id=str(uuid.uuid4()),
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-26T11:00:00+00:00",  # Saturday
        evidence=None,
        trade_proposal=None,
        approved_trade={
            "trade": {
                "id": str(uuid.uuid4()),
                "cycle_id": str(uuid.uuid4()),
                "symbol": "NVDA",
                "side": "buy",
                "qty": "10",
                "status": "proposed",
                "requested_price": "100",
                "fill_price": None,
                "slippage": None,
                "commission": None,
                "idempotency_key": "test-key",
            },
            "correlation_id": "test-cid",
        },
        hitl_status=None,
        cycle_outcome=None,
        error=None,
        token_count=0,
    )

    result = execution_fn(closed_state)

    assert result.get("cycle_outcome") == CycleOutcome.REJECTED_MARKET_CLOSED, (
        f"Expected rejected_market_closed, got {result.get('cycle_outcome')!r}"
    )
    # Ledger is None — if execution had proceeded it would have raised AttributeError.
    # Getting here means no ledger write was attempted.


# ---------------------------------------------------------------------------
# Token-budget wrapper — LLMError returned when budget exceeded (R5 §2)
# ---------------------------------------------------------------------------


def test_token_budget_llm_returns_error_when_over_budget() -> None:
    """Once the per-cycle token budget is exceeded the wrapper returns LLMError.

    Steps:
    1. Create a TokenBudgetLLM with a tight budget (100 tokens).
    2. Set the tracing context var to a known correlation_id.
    3. Record tokens that exceed the budget.
    4. The next complete() call must return LLMError without touching the inner LLM.
    5. The breaker total must reflect only the tokens recorded in step 3.

    Requirement: R5 §2 — token-budget circuit breaker.
    """
    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.domain.guardrails import TokenBudgetCircuitBreaker
    from firm.observability.tracing import reset_correlation_id, set_correlation_id
    from firm.ports.types import LLMError, LLMMessage, LLMResponse

    cid = "test-cycle-budget-" + str(uuid.uuid4())
    token = set_correlation_id(cid)
    try:
        breaker = TokenBudgetCircuitBreaker()
        budget = 100

        canned = LLMResponse(content="ok", input_tokens=10, output_tokens=5, model="fake")
        inner = FakeLLM(responses=[canned] * 10)
        sink = FakeReportSink()

        wrapper = TokenBudgetLLM(inner=inner, breaker=breaker, budget=budget, report_sink=sink)

        # Record tokens exceeding the budget before the first call.
        breaker.record_tokens(cid, 150)  # 150 > 100

        result = wrapper.complete(
            messages=[LLMMessage(role="user", content="hello")],
            model="claude-haiku-4-5",
            max_tokens=256,
        )

        assert isinstance(result, LLMError), (
            f"Expected LLMError when over budget, got {type(result).__name__!r}"
        )
        assert "token budget exceeded" in result.message
        # Breaker total must still be 150 (no new tokens from the blocked call).
        assert breaker.get_total(cid) == 150
        # An alert must have been sent.
        assert len(sink.alerts) == 1
        assert cid in sink.alerts[0][0] or cid in sink.alerts[0][1]
    finally:
        reset_correlation_id(token)


def test_token_budget_llm_records_tokens_on_success() -> None:
    """Tokens from a successful call are recorded in the breaker."""
    from firm.adapters.llm_token_budget import TokenBudgetLLM
    from firm.domain.guardrails import TokenBudgetCircuitBreaker
    from firm.observability.tracing import reset_correlation_id, set_correlation_id
    from firm.ports.types import LLMMessage, LLMResponse

    cid = "test-cycle-record-" + str(uuid.uuid4())
    token = set_correlation_id(cid)
    try:
        breaker = TokenBudgetCircuitBreaker()
        canned = LLMResponse(content="ok", input_tokens=30, output_tokens=20, model="fake")
        inner = FakeLLM(responses=[canned])

        wrapper = TokenBudgetLLM(inner=inner, breaker=breaker, budget=50_000)

        wrapper.complete(
            messages=[LLMMessage(role="user", content="hello")],
            model="claude-haiku-4-5",
            max_tokens=256,
        )

        assert breaker.get_total(cid) == 50  # 30 + 20
    finally:
        reset_correlation_id(token)
