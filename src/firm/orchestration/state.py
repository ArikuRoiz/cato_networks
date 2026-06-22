"""GraphState TypedDict — the lean handoff state threaded through every node.

All values must be JSON-serialisable so the Postgres checkpointer can persist
and restore them without loss.  Complex domain objects are passed as ``dict``
(serialised at the agent boundary); the orchestration layer never imports
domain entities directly.
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """State envelope shared across all pipeline nodes.

    Fields are optional (``total=False``) so each node can return a partial
    dict and LangGraph merges it with the accumulated state.

    Field contract:
    - ``correlation_id``: UUID string, set by the trigger, immutable.
    - ``trigger_type``: ``"scheduled"`` or ``"event"``.
    - ``symbol``: ticker, e.g. ``"NVDA"``.
    - ``decision_ts``: ISO-8601 datetime string (UTC), no-lookahead boundary.
    - ``evidence``: serialised evidence dict from ResearchAgent.
    - ``technical_signal``: serialised TechnicalSignal or TechnicalUnavailable dict.
    - ``trade_proposal``: serialised proposal from PMAgent.
    - ``approved_trade``: copy of ``trade_proposal`` after risk gate passes.
    - ``hitl_status``: lifecycle of the human decision.
    - ``cycle_outcome``: final result of the cycle, written by the last node.
    - ``synthesis``: serialised SynthesisReport or SynthesisFailure dict.
    - ``verdict``: serialised Verdict dict from JudgeAgent.
    - ``error``: human-readable error message if the cycle fails.
    - ``token_count``: running LLM token count, enforced against the budget.
    """

    correlation_id: str
    trigger_type: str  # "scheduled" | "event"
    symbol: str
    decision_ts: str  # ISO datetime string
    evidence: dict | None  # type: ignore[type-arg]
    technical_signal: dict | None  # type: ignore[type-arg]
    trade_proposal: dict | None  # type: ignore[type-arg]
    approved_trade: dict | None  # type: ignore[type-arg]
    hitl_status: str | None  # "pending" | "approved" | "rejected" | "expired"
    cycle_outcome: str | None  # "filled" | "rejected" | "rejected_timeout" | "hold" | "error"
    synthesis: dict | None  # type: ignore[type-arg]
    verdict: dict | None  # type: ignore[type-arg]
    error: str | None
    token_count: int
