"""LangGraph pipeline graph factory.

``build_graph`` wires the five pipeline nodes into a directed graph with a
conditional edge at the risk gate and compiles it with the supplied
checkpointer.

Graph topology:
    START → research → pm → risk → (conditional) → execution → reporting → END
                                 ↘ (rejected)      → reporting → END

The risk node uses ``interrupt()`` internally (not ``interrupt_before``) so the
interrupt site is inside the node body, giving the node full control over the
HITL branching logic after resume.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from firm.config.settings import RiskPolicyConfig, load_risk_policy
from firm.orchestration.nodes import (
    execution_node,
    make_risk_node,
    pm_node,
    reporting_node,
    research_node,
)
from firm.orchestration.state import GraphState

# ---------------------------------------------------------------------------
# Conditional routing after the risk gate
# ---------------------------------------------------------------------------


def _route_after_risk(state: GraphState) -> str:
    """Return the next node name based on the risk gate decision.

    - When ``approved_trade`` is set the trade passed (or was approved by a
      human) → proceed to execution.
    - Any other outcome (rejected, rejected_timeout, error) → skip execution
      and go straight to reporting so the cycle still records an outcome.
    """
    if state.get("approved_trade") is not None:
        return "execution"
    return "reporting"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
    risk_policy: RiskPolicyConfig | None = None,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Construct and compile the LangGraph decision pipeline.

    The *risk_policy* is loaded once here and injected into the risk node via
    :func:`make_risk_node`, so that no file I/O happens during node execution.

    Args:
        checkpointer: A ready-to-use checkpoint saver (e.g. from
            :func:`firm.orchestration.checkpointer.setup_checkpointer`).
            Must be provided — the interrupt/resume mechanism requires it.
        risk_policy: A validated :class:`RiskPolicyConfig`.  When ``None``
            (the default) the policy is loaded from ``config/risk_policy.yaml``
            exactly once at graph-build time.

    Returns:
        A compiled :class:`CompiledStateGraph` ready for ``.invoke()`` /
        ``.stream()`` calls.
    """
    resolved_policy = risk_policy if risk_policy is not None else load_risk_policy()

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]

    builder.add_node("research", research_node)
    builder.add_node("pm", pm_node)
    builder.add_node("risk", make_risk_node(resolved_policy))  # type: ignore[arg-type]
    builder.add_node("execution", execution_node)
    builder.add_node("reporting", reporting_node)

    builder.add_edge(START, "research")
    builder.add_edge("research", "pm")
    builder.add_edge("pm", "risk")
    builder.add_conditional_edges("risk", _route_after_risk, ["execution", "reporting"])
    builder.add_edge("execution", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)
