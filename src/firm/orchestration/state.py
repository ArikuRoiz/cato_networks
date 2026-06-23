"""GraphState TypedDict — the lean handoff state threaded through every node.

All values must be JSON-serialisable so the Postgres checkpointer can persist
and restore them without loss.  Complex domain objects are passed as ``dict``
(serialised at the agent boundary); the orchestration layer never imports
domain entities directly.
"""

from __future__ import annotations

from typing import Any, TypedDict


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
    - ``bull_history``: list of bull researcher argument strings (one per debate round).
    - ``bear_history``: list of bear researcher argument strings (one per debate round).
    - ``debate_rounds``: number of completed bull+bear rounds.
    - ``research_plan``: serialised ResearchPlan from ResearchManagerAgent.
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
    trigger_type: str  # TriggerType: "scheduled" | "event"
    symbol: str
    decision_ts: str  # ISO datetime string
    evidence: dict[str, Any] | None
    technical_signal: dict[str, Any] | None
    bull_history: list[str]
    bear_history: list[str]
    debate_rounds: int
    research_plan: dict[str, Any] | None
    trade_proposal: dict[str, Any] | None
    approved_trade: dict[str, Any] | None
    hitl_status: str | None  # HITLStatus: "pending" | "approved" | "rejected" | "expired"
    hitl_decision: str | None  # HITLDecision: "approve" | "override:buy" | "override:sell" | ...
    cycle_outcome: (
        str | None
    )  # CycleOutcome: "filled" | "rejected" | "rejected_timeout" | "hold" | "error"
    synthesis: dict[str, Any] | None
    verdict: dict[str, Any] | None
    error: str | None
    token_count: int
    force_buy: bool  # Demo/override: inject synthetic BUY plan, skip LLM research-manager call
