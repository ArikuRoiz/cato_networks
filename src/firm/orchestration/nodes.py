"""Pipeline node implementations for the LangGraph decision graph.

Each node is a plain callable that accepts ``GraphState`` and returns a partial-
state ``dict``.  Nodes receive injected ports via ``NodeConfig`` (closed over at
graph-build time) so no IO is constructed inside a running node.

HITL design (risk_node):
    The ``interrupt()`` call in ``risk_node`` follows the LangGraph re-execution
    model: on first entry the call raises ``GraphInterrupt`` (checkpointing the
    state), halting the cycle.  When a client resumes via
    ``Command(resume=hitl_value, update={"hitl_status": ...})`` the node
    re-executes from the top; the ``interrupt()`` call this time returns the
    resume value immediately, so the node can inspect ``hitl_status`` from the
    *updated* state and route accordingly.

Dependency injection:
    Each ``make_*_node`` factory closes over all required ports/agents so that
    construction happens once at graph-build time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from firm.agents.execution import ExecutionAgent, ExecutionInput
from firm.agents.portfolio_manager import PMInput, PortfolioManagerAgent
from firm.agents.reporting import ReportingAgent, ReportingInput
from firm.agents.research import ResearchAgent, ResearchInput
from firm.agents.risk import ApprovedTrade, RiskAgent, RiskInput
from firm.agents.risk import HITLRequired as AgentHITLRequired
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.state import GraphState
from firm.persistence.ledger import LedgerRepository
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.ports.report import ReportSink

# ---------------------------------------------------------------------------
# Port container (injected at graph-build time)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodePorts:
    """All external dependencies needed by the pipeline nodes.

    Instantiated once at graph-build time and closed over in each node factory.
    """

    evidence: EvidenceStore
    llm: LLM
    market_data: MarketDataSource
    ledger: LedgerRepository
    report_sink: ReportSink
    guardrail: LedgerGuardrail
    injection_guard: InjectionGuard
    risk_policy: RiskPolicyConfig
    portfolio_id: UUID
    portfolio: Portfolio


# ---------------------------------------------------------------------------
# Research node
# ---------------------------------------------------------------------------


def make_research_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``research_node`` closed over injected ports."""
    agent = ResearchAgent(
        evidence=ports.evidence,
        llm=ports.llm,
        injection_guard=ports.injection_guard,
    )

    def research_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        correlation_id = state.get("correlation_id", "")
        decision_ts = _parse_datetime(decision_ts_str)
        inp = ResearchInput(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
        )
        result = agent.run(inp)
        return {"evidence": result.model_dump(mode="json")}

    return research_node


# ---------------------------------------------------------------------------
# Portfolio Manager node
# ---------------------------------------------------------------------------


def make_pm_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``pm_node`` closed over injected ports."""
    agent = PortfolioManagerAgent(
        market_data=ports.market_data,
        risk=ports.risk_policy,
    )

    def pm_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        correlation_id = state.get("correlation_id", "")
        decision_ts = _parse_datetime(decision_ts_str)
        evidence_raw = state.get("evidence")
        evidence = _deserialise_evidence(evidence_raw)
        inp = PMInput(
            symbol=symbol,
            evidence=evidence,
            portfolio=ports.portfolio,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
        )
        result = agent.run(inp)
        return {"trade_proposal": result.model_dump(mode="json")}

    return pm_node


# ---------------------------------------------------------------------------
# Risk node — HITL interrupt/resume logic (production-ready)
# ---------------------------------------------------------------------------


def make_risk_node(
    risk_policy: RiskPolicyConfig,
    ports: NodePorts | None = None,
) -> Callable[[GraphState, RunnableConfig], dict[str, Any]]:
    """Return a ``risk_node`` closed over the policy (and optionally ports).

    When *ports* is supplied the real RiskAgent is used; otherwise the node
    still enforces HITL but calls the domain check directly.
    """
    agent = RiskAgent(risk=risk_policy)

    def risk_node(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        proposal_raw = state.get("trade_proposal")
        if proposal_raw is None:
            return {"cycle_outcome": "error", "error": "trade_proposal missing in risk_node"}

        proposal = _deserialise_proposal(proposal_raw)
        if ports is None:
            # WARNING: no NodePorts supplied — risk limits are evaluated against
            # a synthetic $10,000 NAV.  Supply NodePorts.portfolio in production
            # so HITL/rejection thresholds reflect the real portfolio NAV.
            portfolio = _empty_portfolio()
        else:
            portfolio = ports.portfolio
        prices = _extract_prices(proposal, portfolio)
        correlation_id = state.get("correlation_id", "")

        inp = RiskInput(
            proposal=proposal,
            portfolio=portfolio,
            prices=prices,
            correlation_id=correlation_id,
        )
        risk_result = agent.run(inp)

        # HITL path: use LangGraph interrupt so the checkpointer can persist
        # state before the node yields to a human reviewer.
        if isinstance(risk_result, AgentHITLRequired):
            # Pre-build the ApprovedTrade so idempotency_key/UUID are stable
            # across the interrupt boundary.  The serialised form is passed to
            # the interrupt payload so the resume handler can reconstruct it.
            pending_approved = _build_approved_from_hitl(risk_result, correlation_id)
            interrupt(
                {
                    "type": "hitl_request",
                    "trade_proposal": proposal_raw,
                    "approved_trade": pending_approved.model_dump(mode="json"),
                }
            )
            # Re-entry after resume: hitl_status is in state via Command(update=...).
            hitl_status = state.get("hitl_status")
            return _route_hitl(hitl_status, pending_approved)

        return _map_risk_result(risk_result, proposal_raw)

    return risk_node


def _map_risk_result(risk_result: object, proposal_raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a RiskAgent result to a graph-state update dict.

    HITLRequired is handled before this function is called (via interrupt()).
    When the trade is approved we serialise the full ``ApprovedTrade`` (including
    the risk-checked ``Trade`` object with its idempotency_key and UUID) so that
    the execution node can rehydrate it directly without constructing a new Trade.
    """
    from firm.agents.risk import Rejected

    if isinstance(risk_result, ApprovedTrade):
        return {"approved_trade": risk_result.model_dump(mode="json")}
    if isinstance(risk_result, Rejected):
        return {"cycle_outcome": "rejected"}
    return {"cycle_outcome": "error", "error": f"unexpected risk result: {risk_result!r}"}


def _route_hitl(hitl_status: str | None, pending_approved: ApprovedTrade) -> dict[str, Any]:
    """Map hitl_status to graph state after HITL resume.

    *pending_approved* is the ``ApprovedTrade`` built before the interrupt so
    that the same Trade object (same UUID + idempotency_key) is used whether
    the human approves immediately or after a delay.
    """
    if hitl_status == "rejected":
        return {"cycle_outcome": "rejected"}
    if hitl_status == "expired":
        return {"cycle_outcome": "rejected_timeout"}
    if hitl_status == "approved":
        return {"approved_trade": pending_approved.model_dump(mode="json")}
    return {
        "cycle_outcome": "error",
        "error": f"unexpected hitl_status after resume: {hitl_status!r}",
    }


def _hitl_exceeds_threshold(notional: Decimal, threshold: Decimal) -> bool:
    """Return True when *notional* strictly exceeds the HITL *threshold*.

    The boundary is strictly-above (``>``): a notional equal to the threshold
    does **not** trigger HITL.
    """
    return notional > threshold


def _build_approved_from_hitl(
    hitl_result: AgentHITLRequired,
    correlation_id: str,
) -> ApprovedTrade:
    """Build an ApprovedTrade from a HITLRequired result.

    Constructs the Trade domain object using the same logic as RiskAgent so the
    idempotency_key is stable across the interrupt boundary.
    """
    from firm.agents.risk import _build_trade_stub

    proposal = hitl_result.proposal
    prices: dict[str, Decimal] = {
        proposal.symbol: proposal.notional / proposal.qty
        if proposal.qty > Decimal("0")
        else Decimal("0")
    }
    trade = _build_trade_stub(proposal, prices, correlation_id)
    return ApprovedTrade(trade=trade, correlation_id=correlation_id)


# ---------------------------------------------------------------------------
# Execution node
# ---------------------------------------------------------------------------


def make_execution_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return an ``execution_node`` closed over injected ports."""
    agent = ExecutionAgent(ledger=ports.ledger, guardrail=ports.guardrail)

    def execution_node(state: GraphState) -> dict[str, Any]:
        from firm.agents.execution import Fill

        approved_raw = state.get("approved_trade")
        if approved_raw is None:
            return {"cycle_outcome": "error", "error": "approved_trade missing in execution_node"}

        correlation_id = state.get("correlation_id", "")
        approved = _deserialise_approved_trade(approved_raw)
        if approved is None:
            return {"cycle_outcome": "error", "error": "approved_trade could not be deserialised"}

        trade = approved.trade
        prices = {trade.symbol: trade.requested_price}
        inp = ExecutionInput(
            approved_trade=approved,
            portfolio_id=ports.portfolio_id,
            portfolio=ports.portfolio,
            prices=prices,
            correlation_id=correlation_id,
        )
        result = agent.run(inp)
        if isinstance(result, Fill):
            return {"cycle_outcome": "filled"}
        return {"cycle_outcome": "error", "error": result.reason}

    return execution_node


# ---------------------------------------------------------------------------
# Reporting node
# ---------------------------------------------------------------------------


def make_reporting_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``reporting_node`` closed over injected ports."""
    agent = ReportingAgent(report_sink=ports.report_sink, ledger=ports.ledger)

    def reporting_node(state: GraphState) -> dict[str, Any]:
        from firm.agents.reporting import ReportFailure

        correlation_id = state.get("correlation_id", "")
        decision_ts_str = state.get("decision_ts", "")
        decision_ts = _parse_datetime(decision_ts_str)
        cycle_id = _str_to_uuid(correlation_id)
        inp = ReportingInput(
            cycle_id=cycle_id,
            portfolio_id=ports.portfolio_id,
            report_date=decision_ts.date(),
            correlation_id=correlation_id,
        )
        result = agent.run(inp)
        outcome = state.get("cycle_outcome", "filled")
        if isinstance(result, ReportFailure):
            return {"cycle_outcome": outcome}  # degrade gracefully; don't overwrite outcome
        return {"cycle_outcome": outcome}

    return reporting_node


# ---------------------------------------------------------------------------
# Fallback stubs (used when NodePorts is not provided; for backward compat)
# ---------------------------------------------------------------------------


def research_node(state: GraphState) -> dict[str, Any]:
    """Pass-through stub — replaced by make_research_node in production."""
    return {"evidence": {"chunks": [], "summary": "stub"}}


def pm_node(state: GraphState) -> dict[str, Any]:
    """Pass-through stub — replaced by make_pm_node in production."""
    symbol = state.get("symbol", "")
    return {
        "trade_proposal": {
            "symbol": symbol,
            "side": "buy",
            "qty": "10",
            "notional": "1000",
            "rationale": "stub",
        }
    }


def execution_node(state: GraphState) -> dict[str, Any]:
    """Pass-through stub — replaced by make_execution_node in production."""
    return {"cycle_outcome": "filled"}


def reporting_node(state: GraphState) -> dict[str, Any]:
    """Pass-through stub — replaced by make_reporting_node in production."""
    return {"cycle_outcome": state.get("cycle_outcome", "filled")}


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime string; fall back to now on failure."""
    from datetime import UTC

    if not value:
        return datetime.now(tz=UTC)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(tz=UTC)


def _str_to_uuid(value: str) -> UUID:
    """Parse *value* as UUID; generate a fresh one on failure."""
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return uuid4()


def _deserialise_evidence(raw: dict[str, Any] | None) -> Any:
    """Deserialise evidence dict to Evidence or Refusal."""
    from firm.agents.research import Evidence, Refusal

    if raw is None:
        return Refusal(reason="insufficient_evidence")
    if "claims" in raw:
        try:
            return Evidence.model_validate(raw)
        except Exception:
            return Refusal(reason="insufficient_evidence")
    if "reason" in raw:
        try:
            return Refusal.model_validate(raw)
        except Exception:
            return Refusal(reason="insufficient_evidence")
    return Refusal(reason="insufficient_evidence")


def _deserialise_proposal(raw: dict[str, Any]) -> Any:
    """Deserialise proposal dict to TradeProposal or Hold."""
    from firm.agents.portfolio_manager import Hold, TradeProposal

    if "qty" in raw and "notional" in raw:
        try:
            return TradeProposal.model_validate(raw)
        except Exception:
            pass
    if "reason" in raw:
        try:
            return Hold.model_validate(raw)
        except Exception:
            pass
    symbol = raw.get("symbol", "")
    return Hold(symbol=symbol, reason="deserialisation failed")


def _deserialise_approved_trade(raw: dict[str, Any]) -> ApprovedTrade | None:
    """Deserialise an ``approved_trade`` state value to an ``ApprovedTrade``.

    Returns ``None`` on any parse failure so callers can return a typed error
    state rather than propagating an exception.
    """
    try:
        return ApprovedTrade.model_validate(raw)
    except Exception:
        return None


def _extract_prices(proposal: object, portfolio: object) -> dict[str, Decimal]:
    """Extract a minimal prices dict for risk evaluation.

    Derives the proposal's implied price from notional/qty and uses avg_cost
    as a fallback for existing holdings when no live prices are available.
    """
    from firm.agents.portfolio_manager import TradeProposal
    from firm.domain import Portfolio as PortfolioModel

    prices: dict[str, Decimal] = {}
    if isinstance(proposal, TradeProposal) and proposal.qty > Decimal("0"):
        prices[proposal.symbol] = proposal.notional / proposal.qty
    if isinstance(portfolio, PortfolioModel):
        for sym, holding in portfolio.holdings.items():
            if sym not in prices:
                prices[sym] = holding.avg_cost
    return prices


def _empty_portfolio() -> Portfolio:
    """Return a synthetic portfolio for contexts where no real portfolio is available.

    WARNING: risk limits (HITL threshold, max notional) are evaluated against this
    $10,000 NAV, not the real portfolio.  Always supply ``NodePorts.portfolio`` in
    production via ``make_risk_node(policy, ports=node_ports)``.
    """
    return Portfolio(cash=Decimal("10000"))
