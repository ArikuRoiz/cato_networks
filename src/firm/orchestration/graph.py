"""LangGraph pipeline graph factory.

Graph topology:
    START → research ──┐
    START → technical ─┴→ debate_bull → debate_bear ─┬→ debate_bull (loop if rounds < MAX)
                                                      └→ research_manager → pm → risk
                         → (approved) execution → reporting → synthesis → judge → END
                         → (rejected) reporting → synthesis → judge → END

research and technical run in parallel; debate_bull fans in from both.
Bull and bear alternate MAX_DEBATE_ROUNDS times, then research_manager adjudicates.
The adjudicated ResearchPlan feeds PM as the primary sentiment signal.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from firm.config.settings import RiskPolicyConfig, load_risk_policy
from firm.orchestration.nodes import (
    MAX_DEBATE_ROUNDS,
    NodePorts,
    make_bear_node,
    make_bull_node,
    make_execution_node,
    make_judge_node,
    make_pm_node,
    make_reporting_node,
    make_research_manager_node,
    make_research_node,
    make_risk_node,
    make_synthesis_node,
    make_technical_node,
)
from firm.orchestration.state import GraphState

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def _route_after_bear(state: GraphState) -> str:
    rounds = state.get("debate_rounds", 0)
    if rounds >= MAX_DEBATE_ROUNDS:
        return "research_manager"
    return "debate_bull"


def _route_after_risk(state: GraphState) -> str:
    if state.get("approved_trade") is not None:
        return "execution"
    return "reporting"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
    risk_policy: RiskPolicyConfig | None = None,
    ports: NodePorts | None = None,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    resolved_policy = risk_policy if risk_policy is not None else load_risk_policy()
    if ports is None:
        raise ValueError("NodePorts must be supplied — build_graph() requires a fully wired ports container")

    builder: StateGraph = StateGraph(GraphState)  # type: ignore[type-arg]

    builder.add_node("research", make_research_node(ports))
    builder.add_node("technical", make_technical_node(ports))
    builder.add_node("debate_bull", make_bull_node(ports))
    builder.add_node("debate_bear", make_bear_node(ports))
    builder.add_node("research_manager", make_research_manager_node(ports))
    builder.add_node("pm", make_pm_node(ports))
    builder.add_node("risk", make_risk_node(resolved_policy, ports))
    builder.add_node("execution", make_execution_node(ports))
    builder.add_node("reporting", make_reporting_node(ports))
    builder.add_node("synthesis", make_synthesis_node(ports))
    builder.add_node("judge", make_judge_node(ports))

    # Parallel fan-out; debate_bull waits for both research and technical
    builder.add_edge(START, "research")
    builder.add_edge(START, "technical")
    builder.add_edge("research", "debate_bull")
    builder.add_edge("technical", "debate_bull")

    # Debate loop — bear routes back to bull or exits to research_manager
    builder.add_edge("debate_bull", "debate_bear")
    builder.add_conditional_edges(
        "debate_bear", _route_after_bear, ["debate_bull", "research_manager"]
    )

    # Main pipeline after debate
    builder.add_edge("research_manager", "pm")
    builder.add_edge("pm", "risk")
    builder.add_conditional_edges("risk", _route_after_risk, ["execution", "reporting"])
    builder.add_edge("execution", "reporting")
    builder.add_edge("reporting", "synthesis")
    builder.add_edge("synthesis", "judge")
    builder.add_edge("judge", END)

    return builder.compile(checkpointer=checkpointer)
