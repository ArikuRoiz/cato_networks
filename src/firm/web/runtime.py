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

import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

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


def build_live_graph(settings: Any) -> LiveGraph:
    """Wire all live adapters and return a ready-to-use graph + portfolio_id.

    Delegates the heavy lifting to the shared composition root
    (:func:`firm.composition.build_live_pipeline`) and adapts the returned
    :class:`firm.composition.Pipeline` into the web-specific :class:`LiveGraph`,
    which exposes the checkpointer, engine and ledger that the HITL endpoints
    and pending-run registry need.  ``ensure_portfolio`` is already called
    inside the composition builder, so GET /api/portfolio always finds a row
    with the real starting cash.
    """
    from firm.composition import build_live_pipeline

    pipeline = build_live_pipeline(settings)
    return LiveGraph(
        graph=pipeline.graph,
        portfolio_id=pipeline.portfolio_id,
        checkpointer=pipeline.checkpointer,
        engine=pipeline.engine,
        ledger=pipeline.ledger,
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
        logger.exception("pending_approvals scan failed; returning no pending threads")
        return []


def resume_approval(
    live_graph: LiveGraph,
    thread_id: str,
    decision: str,
    edited_qty: Decimal | None,
) -> dict[str, Any]:
    """Resume an interrupted graph thread with the operator decision.

    Builds a LangGraph ``Command`` equivalent to what the CLI does in
    ``_invoke_live_symbol`` after the approval channel returns a decision.  Also
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
        logger.debug("get_state failed for thread %s; skipping", thread_id, exc_info=True)
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
        logger.warning("Failed to unregister pending run %s", thread_id, exc_info=True)


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
        logger.exception("Background cycle for %s (thread %s) failed", symbol, thread_id)
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
        logger.warning("Failed to register pending run %s", thread_id, exc_info=True)
