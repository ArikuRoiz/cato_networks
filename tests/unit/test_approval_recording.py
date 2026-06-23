"""Unit tests for HITL approval recording (Ticket R6).

Tests the full recording path without a real database:
  - _FakeLedger.record_approval accumulates ApprovalRecord entries.
  - risk_node calls record_approval on HITL resume (approved, rejected, expired).

These tests are fast (no Postgres, no LLM) and run in the standard unit suite.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from eval.replay import _FakeLedger
from firm.domain import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ledger() -> _FakeLedger:
    return _FakeLedger(portfolio=Portfolio(cash=Decimal("100000")), portfolio_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# _FakeLedger.record_approval
# ---------------------------------------------------------------------------


def test_record_approval_appends_one_record() -> None:
    ledger = _make_ledger()
    correlation_id = uuid.uuid4()
    trade_id = uuid.uuid4()

    ledger.record_approval(
        correlation_id=correlation_id,
        trade_id=trade_id,
        status="approved",
        original_notional=Decimal("1000"),
        original_qty=Decimal("10"),
    )

    assert len(ledger.approvals) == 1
    rec = ledger.approvals[0]
    assert rec.correlation_id == correlation_id
    assert rec.trade_id == trade_id
    assert rec.status == "approved"
    assert rec.original_notional == Decimal("1000")
    assert rec.original_qty == Decimal("10")
    assert rec.edited_qty is None
    assert rec.decided_by == "risk_committee"
    assert rec.decided_at is not None


def test_record_approval_persists_edited_qty() -> None:
    ledger = _make_ledger()

    ledger.record_approval(
        correlation_id=uuid.uuid4(),
        trade_id=uuid.uuid4(),
        status="edited",
        original_notional=Decimal("2000"),
        original_qty=Decimal("20"),
        edited_qty=Decimal("12"),
        decided_by="head_of_risk",
    )

    rec = ledger.approvals[0]
    assert rec.edited_qty == Decimal("12")
    assert rec.decided_by == "head_of_risk"


def test_record_approval_accumulates_multiple_decisions() -> None:
    ledger = _make_ledger()

    for status in ("approved", "rejected", "expired"):
        ledger.record_approval(
            correlation_id=uuid.uuid4(),
            trade_id=uuid.uuid4(),
            status=status,
            original_notional=Decimal("500"),
            original_qty=Decimal("5"),
        )

    assert len(ledger.approvals) == 3
    statuses = [r.status for r in ledger.approvals]
    assert statuses == ["approved", "rejected", "expired"]


# ---------------------------------------------------------------------------
# risk_node calls record_approval on HITL resume
# ---------------------------------------------------------------------------


def _make_ports_with_fake_ledger(hitl_threshold_pct: float = 0.05) -> object:
    """Build NodePorts with _FakeLedger for HITL recording tests."""
    from firm.adapters.fakes import (
        FakeCalendar,
        FakeEvidenceStore,
        FakeLLM,
        FakeMarketData,
        FakeReportSink,
    )
    from firm.config.settings import RiskPolicyConfig
    from firm.domain import RiskPolicy
    from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
    from firm.orchestration.nodes import NodePorts

    risk_policy = RiskPolicyConfig(
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
        token_budget_per_cycle=50000,
    )
    domain_policy = RiskPolicy(
        max_trade_notional_pct=Decimal(str(hitl_threshold_pct)),
        max_name_concentration_pct=Decimal("0.25"),
        daily_loss_halt_pct=Decimal("0.03"),
        hitl_threshold_pct=Decimal(str(hitl_threshold_pct)),
    )
    portfolio = Portfolio(cash=Decimal("10000"))
    ledger = _FakeLedger(portfolio=portfolio, portfolio_id=uuid.uuid4())
    ports = NodePorts(
        evidence=FakeEvidenceStore(),
        llm=FakeLLM(),
        market_data=FakeMarketData(),
        ledger=ledger,  # type: ignore[arg-type]
        report_sink=FakeReportSink(),
        guardrail=LedgerGuardrail(domain_policy),
        injection_guard=InjectionGuard(),
        risk_policy=risk_policy,
        portfolio_id=uuid.uuid4(),
        portfolio=portfolio,
        calendar=FakeCalendar(is_open=True),
    )
    return ports, ledger


@pytest.mark.parametrize("hitl_status", ["approved", "rejected", "expired"])
def test_risk_node_resume_records_hitl_decision(hitl_status: str) -> None:
    """Resuming the HITL interrupt path in risk_node does not write an ApprovalRow.

    Design change: ApprovalRow persistence is deferred out of risk_node to avoid
    a FK violation (the TradeRow does not exist yet at interrupt time):

      - approved  → ApprovalRow is written by execution_node AFTER ledger.buy()
      - rejected  → no ApprovalRow (only a log line); cycle record is the audit trail
      - expired   → same as rejected

    This test verifies that risk_node does NOT write an approval in any of the
    three cases, preserving the risk_node's role as interrupt-and-route-only.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph
    from langgraph.types import Command

    from firm.orchestration.nodes import make_risk_node
    from firm.orchestration.state import GraphState

    ports, ledger = _make_ports_with_fake_ledger(hitl_threshold_pct=0.05)
    risk_fn = make_risk_node(ports)

    def _inject(state: GraphState) -> dict:  # type: ignore[type-arg]
        return {}

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]
    builder.add_node("inject", _inject)
    builder.add_node("risk", risk_fn)
    builder.add_edge(START, "inject")
    builder.add_edge("inject", "risk")
    builder.add_edge("risk", END)
    saver = MemorySaver()
    mini_graph = builder.compile(checkpointer=saver)

    correlation_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    initial_state = GraphState(
        correlation_id=correlation_id,
        trigger_type="scheduled",
        symbol="NVDA",
        decision_ts="2024-10-23T10:00:00+00:00",
        evidence=None,
        # notional=1000 > threshold=500 (5% of NAV=10000)
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
    config = {"configurable": {"thread_id": thread_id}}

    # First pass: interrupt is raised, no approval recorded yet.
    events = list(mini_graph.stream(initial_state, config, stream_mode="updates"))
    interrupt_events = [e for e in events if "__interrupt__" in e]
    assert interrupt_events, "Expected HITL interrupt"
    assert len(ledger.approvals) == 0, "No approval must be written before resume"

    # Resume with the given hitl_status.
    resume_cmd = Command(
        resume={"decision": hitl_status},
        update={"hitl_status": hitl_status},
    )
    list(mini_graph.stream(resume_cmd, config, stream_mode="updates"))

    # risk_node must NOT write an ApprovalRow — see docstring for the three cases.
    assert len(ledger.approvals) == 0, (
        f"risk_node must not write an ApprovalRow (status={hitl_status!r}); "
        "approved → deferred to execution_node; rejected/expired → log only."
    )
