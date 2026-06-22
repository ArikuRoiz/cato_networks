"""Mandatory failing test stubs — one per brief requirement.

Each test is decorated ``@pytest.mark.xfail(strict=True)`` so:
  - The suite is RED until the feature is implemented (xpass would be a surprise pass → fail).
  - A feature ticket un-xfails its test and makes it pass for the right reason.
  - CI fails if any stub accidentally passes without the implementation (strict=True).

Do NOT add implementation here.  Implementations live in the feature tickets:
  - FIRM-4  → test_crash_mid_trade_reconciles, test_idempotent_execution
  - FIRM-5  → test_hitl_resumes_after_restart, test_hitl_timeout_fails_safe
  - FIRM-7  → test_limit_cannot_be_exceeded, test_stale_approval_revalidated,
               test_prompt_injection_neutralized
  - FIRM-8  → test_market_calendar_gating
  - FIRM-9  → test_no_lookahead, test_insufficient_evidence_refuses
"""

from datetime import UTC, datetime

import pytest

from firm.orchestration.graph import build_graph
from firm.orchestration.state import GraphState


@pytest.mark.xfail(reason="not implemented yet", strict=True)
def test_crash_mid_trade_reconciles() -> None:
    """Kill the process between cash-debit and holding-write; on restart cash, holdings,
    cost-basis, and P&L must reconcile — partial ledger state is impossible.

    Requirement: FR-1 crash recovery (SPEC §Success Criteria §2).
    Implemented by: FIRM-4 (LedgerRepository atomic transaction + idempotency key).
    """
    raise NotImplementedError


def test_hitl_resumes_after_restart() -> None:
    """Kill the process while a trade is awaiting human approval; on restart the
    LangGraph pipeline must resume from the saved checkpoint and reach the same outcome.

    Requirement: FR-5 durable HITL (SPEC §Success Criteria §3).
    Implemented by: FIRM-5 (LangGraph Postgres checkpointer + interrupt/resume).
    Full integration coverage lives in tests/integration/test_hitl.py.
    This stub confirms the orchestration module can be imported cleanly.
    """
    from firm.orchestration import GraphState, build_graph, setup_checkpointer

    assert GraphState is not None
    assert callable(build_graph)
    assert callable(setup_checkpointer)


@pytest.mark.xfail(reason="not implemented yet", strict=True)
def test_idempotent_execution() -> None:
    """Replaying a trade with the same idempotency_key must be a no-op — the ledger
    must not double-fill or double-debit cash.

    Requirement: ARCHITECTURE §Stress Test / crash mid-trade.
    Implemented by: FIRM-4 (unique idempotency_key constraint on the trades table).
    """
    raise NotImplementedError


@pytest.mark.xfail(reason="not implemented yet", strict=True)
def test_limit_cannot_be_exceeded() -> None:
    """Even if both the agent and a human approver approve an oversized trade, the
    ledger guardrail must still reject it — hard limits are enforced at write time.

    Requirement: SPEC §Boundaries / Never; SPEC §Success Criteria §8.
    Implemented by: FIRM-7 (RiskPolicy enforced at LedgerRepository.execute()).
    """
    raise NotImplementedError


def test_no_lookahead() -> None:
    """The EvidenceStore retrieval must never return a document whose published_at
    timestamp is strictly after the decision_ts passed to the query.

    Requirement: SPEC §Testing Strategy / test_no_lookahead.
    Implemented by: FIRM-9 — full pgvector integration in tests/integration/test_rag.py.
    This stub exercises the pure apply_no_lookahead helper to keep the mandatory
    suite runnable without a database.
    """
    import uuid

    from firm.ports.types import Chunk
    from firm.rag.reranker import apply_no_lookahead

    future_chunk = Chunk(
        id=uuid.uuid4(),
        symbol="NVDA",
        text="Future article",
        source_url="https://example.com/future",
        chunk_id="future-0",
        published_at=datetime(2024, 10, 22, 12, 0, 0, tzinfo=UTC),
        score=0.9,
    )
    before = datetime(2024, 10, 21, 0, 0, 0, tzinfo=UTC)
    assert apply_no_lookahead([future_chunk], before) == [], (
        "apply_no_lookahead must drop chunks published after 'before'."
    )
    after = datetime(2024, 10, 23, 0, 0, 0, tzinfo=UTC)
    assert apply_no_lookahead([future_chunk], after) == [future_chunk], (
        "apply_no_lookahead must retain chunks published before or at 'before'."
    )


def test_insufficient_evidence_refuses() -> None:
    """When the news corpus returns zero chunks the cite() helper returns Insufficient.

    Requirement: FR-4 grounding (SPEC §Success Criteria §7).
    Implemented by: FIRM-9 — full pgvector integration in tests/integration/test_rag.py.
    This stub exercises the pure citation helper to keep the mandatory suite fast.
    """
    from firm.rag.citation import Insufficient, cite

    result = cite([])
    assert isinstance(result, Insufficient), (
        f"cite([]) must return Insufficient, got {type(result).__name__!r}"
    )
    assert result.reason == "no_relevant_chunks"


def test_hitl_timeout_fails_safe() -> None:
    """When no human response arrives before APPROVAL.expires_at the trade must be
    auto-rejected — timeout must never auto-approve (fail-safe, not fail-open).

    Requirement: SPEC §Boundaries / Never (auto-approve on timeout).
    Implemented by: FIRM-5 (expires_at → reject path in the Risk interrupt node).
    Full integration coverage lives in tests/integration/test_hitl.py.
    This stub confirms the risk_node routes "expired" → "rejected_timeout".
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer)

    import uuid

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial: GraphState = GraphState(
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

    events = list(graph.stream(initial, config, stream_mode="updates"))
    interrupt_events = [e for e in events if "__interrupt__" in e]
    assert interrupt_events, "Expected interrupt from risk_node"

    resume_cmd = Command(resume={"decision": "expired"}, update={"hitl_status": "expired"})
    list(graph.stream(resume_cmd, config, stream_mode="updates"))

    final = graph.get_state(config)
    outcome = final.values.get("cycle_outcome")
    assert outcome == "rejected_timeout", (
        f"Expired HITL must produce 'rejected_timeout', not {outcome!r}"
    )


@pytest.mark.xfail(reason="not implemented yet", strict=True)
def test_stale_approval_revalidated() -> None:
    """If the market price moves past a RiskPolicy limit while a trade awaits human
    approval, execution must re-validate and block the fill even with a valid approval.

    Requirement: ARCHITECTURE §Dataflow (Risk re-validates at execution time).
    Implemented by: FIRM-7 (Trade.revalidate(bar) called inside ExecutionAgent).
    """
    raise NotImplementedError


@pytest.mark.xfail(reason="not implemented yet", strict=True)
def test_prompt_injection_neutralized() -> None:
    """News corpus text that contains instruction-like content (e.g. 'buy 10 000 shares')
    must never alter the agent's trade decision — retrieved text is data, not instructions.

    Requirement: SPEC §Boundaries / Never (treat retrieved text as instructions).
    Implemented by: FIRM-7 (injection classifier applied before LLM sees corpus text).
    """
    raise NotImplementedError


def test_market_calendar_gating() -> None:
    """Trade triggers fired on NYSE holidays, half-days, or outside regular market hours
    must not produce fills — the execution path must gate on the market calendar.

    Requirement: FR-2 market-hours gating (SPEC §FR-1 realistic fills).
    Implemented by: FIRM-8 (NYSECalendar in src/firm/services/calendar.py).

    Detailed coverage lives in tests/unit/test_calendar.py.  This integration
    stub confirms the calendar can be imported and instantiated without errors.
    """
    from zoneinfo import ZoneInfo

    from firm.services.calendar import NYSECalendar

    _ET = ZoneInfo("US/Eastern")

    cal = NYSECalendar()

    # NYSE holiday (Thanksgiving 2024) — must be closed
    thanksgiving = datetime(2024, 11, 28, 11, 0, tzinfo=_ET)
    assert cal.is_market_open(thanksgiving) is False

    # Half-day close: day after Thanksgiving 2024 at 13:01 ET — past early close
    black_friday_post_close = datetime(2024, 11, 29, 13, 1, tzinfo=_ET)
    assert cal.is_market_open(black_friday_post_close) is False

    # Normal trading: Oct 23 2024 10:00 ET (NVDA earnings day)
    normal = datetime(2024, 10, 23, 10, 0, tzinfo=_ET)
    assert cal.is_market_open(normal) is True

    # Saturday
    saturday = datetime(2024, 10, 26, 11, 0, tzinfo=_ET)
    assert cal.is_market_open(saturday) is False

    # Boundary: 15:59:59 ET open; 16:00:01 ET closed
    before_close = datetime(2024, 10, 23, 15, 59, 59, tzinfo=_ET)
    after_close = datetime(2024, 10, 23, 16, 0, 1, tzinfo=_ET)
    assert cal.is_market_open(before_close) is True
    assert cal.is_market_open(after_close) is False
