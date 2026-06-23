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

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from firm.agents.debater import DebaterAgent, DebaterFailure, DebaterInput
from firm.agents.execution import ExecutionAgent, ExecutionInput
from firm.agents.judge import JudgeAgent, JudgeFailure, JudgeInput
from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.agents.reporting import ReportingAgent, ReportingInput
from firm.agents.research import ResearchAgent, ResearchInput
from firm.agents.research_manager import (
    ResearchManagerAgent,
    ResearchManagerFailure,
    ResearchManagerInput,
)
from firm.agents.risk import ApprovedTrade, RiskAgent, RiskInput
from firm.agents.risk import HITLRequired as AgentHITLRequired
from firm.agents.synthesis import SynthesisInput, SynthesisReportAgent
from firm.agents.technical import TechnicalAnalysisAgent, TechnicalInput
from firm.config.settings import RiskPolicyConfig
from firm.domain import Portfolio
from firm.domain.enums import CycleOutcome, HITLStatus, Recommendation, RefusalReason, TradeSide
from firm.domain.guardrails import InjectionGuard, LedgerGuardrail
from firm.orchestration.state import GraphState
from firm.persistence.ledger import CycleAuditRecord, LedgerRepository
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.ports.report import ReportSink
from firm.services.calendar import NYSECalendar
from firm.tools.size_position import size_position, trade_side_from_recommendation
from firm.utils import str_to_uuid

logger = logging.getLogger(__name__)

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
    calendar: NYSECalendar


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
# Sizing node (replaces PortfolioManagerAgent — deterministic, no LLM)
# ---------------------------------------------------------------------------


def make_pm_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a deterministic sizing node closed over injected ports.

    Consumes the Research Manager's ``ResearchPlan`` (recommendation + conviction),
    fetches the current bar price via market_data, computes NAV from the portfolio,
    and calls ``size_position`` to produce a ``TradeProposal | Hold``.

    No LLM is invoked here — the Research Manager is the sole direction-decider.
    """

    def pm_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        decision_ts = _parse_datetime(decision_ts_str)
        research_plan = _deserialise_research_plan(state.get("research_plan"))

        if research_plan is None:
            hold = Hold(symbol=symbol, reason="research_plan unavailable")
            return {"trade_proposal": hold.model_dump(mode="json")}

        recommendation: Recommendation = research_plan.recommendation
        conviction: float = research_plan.conviction

        bar = ports.market_data.get_bar(symbol, decision_ts)
        if bar is None:
            hold = Hold(symbol=symbol, reason="no market data for sizing")
            return {"trade_proposal": hold.model_dump(mode="json")}

        prices = {symbol: bar.close}
        for sym, holding in ports.portfolio.holdings.items():
            if sym not in prices:
                prices[sym] = holding.avg_cost
        nav = ports.portfolio.nav(prices)

        qty = size_position(
            recommendation=recommendation,
            conviction=conviction,
            nav=nav,
            price=bar.close,
            max_trade_notional_pct=ports.risk_policy.max_trade_notional_pct,
        )

        if qty < Decimal("1"):
            hold = Hold(
                symbol=symbol,
                reason=f"sizing yielded zero quantity (recommendation={recommendation} conviction={conviction:.3f})",
            )
            return {"trade_proposal": hold.model_dump(mode="json")}

        side = trade_side_from_recommendation(recommendation)
        if side is None:
            hold = Hold(symbol=symbol, reason=f"hold recommendation: {recommendation}")
            return {"trade_proposal": hold.model_dump(mode="json")}

        proposal = TradeProposal(
            symbol=symbol,
            side=TradeSide(side),
            qty=qty,
            notional=qty * bar.close,
            rationale=f"research_manager={recommendation}@{conviction:.2f} nav={float(nav):.0f} price={float(bar.close):.2f}",
        )
        return {"trade_proposal": proposal.model_dump(mode="json")}

    return pm_node


# ---------------------------------------------------------------------------
# Risk node — HITL interrupt/resume logic (production-ready)
# ---------------------------------------------------------------------------


def make_risk_node(
    ports: NodePorts,
) -> Callable[[GraphState, RunnableConfig], dict[str, Any]]:
    agent = RiskAgent(risk=ports.risk_policy)

    def risk_node(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        proposal_raw = state.get("trade_proposal")
        if proposal_raw is None:
            return {
                "cycle_outcome": CycleOutcome.ERROR,
                "error": "trade_proposal missing in risk_node",
            }

        proposal = _deserialise_proposal(proposal_raw)
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
            _record_hitl_decision(ports, hitl_status, pending_approved, risk_result)
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
        return {"cycle_outcome": CycleOutcome.REJECTED}
    return {
        "cycle_outcome": CycleOutcome.ERROR,
        "error": f"unexpected risk result: {risk_result!r}",
    }


def _record_hitl_decision(
    ports: NodePorts,
    hitl_status: str | None,
    pending_approved: ApprovedTrade,
    hitl_required: AgentHITLRequired,
) -> None:
    """Durably record a HITL decision before routing.

    For the approved path, the ApprovalRow is deferred to execution_node so
    the FK constraint on approvals.trade_id is satisfied (the trade row must
    exist first).  For rejected/expired paths, no TradeRow will be created, so
    we write only a warning log — the cycle-outcome audit written by
    ``record_cycle`` already captures the rejection durably.

    Failures are caught and logged — a recording error must never abort the
    decision path.
    """
    if ports.ledger is None:
        return
    if hitl_status == HITLStatus.APPROVED:
        # Approved path: trade hasn't been written yet — defer to execution_node,
        # which calls _record_approval_after_fill_from_state() after ledger.buy().
        return
    # Rejected / expired: no trade row will be created.  Log the decision; the
    # cycle-outcome record (written by the reporting_node via record_cycle) is the
    # durable audit trail.
    logger.info(
        "HITL decision recorded: status=%s correlation_id=%s (not executed)",
        hitl_status,
        pending_approved.correlation_id,
    )


def _record_approval_after_fill_from_state(
    ports: NodePorts,
    approved: ApprovedTrade,
    state: GraphState,
) -> None:
    """Write ApprovalRow + audit entry AFTER the trade is persisted.

    Called by execution_node after a successful fill so the FK constraint on
    approvals.trade_id is satisfied.  Reads original notional/qty from the
    trade_proposal in state (set before the interrupt).  Failures are logged.
    """
    if ports.ledger is None:
        return
    proposal_raw = state.get("trade_proposal") or {}
    original_notional = Decimal(str(proposal_raw.get("notional", "0")))
    original_qty = Decimal(str(proposal_raw.get("qty", "0")))
    try:
        ports.ledger.record_approval(
            correlation_id=uuid.UUID(approved.correlation_id),
            trade_id=approved.trade.id,
            status=HITLStatus.APPROVED,
            original_notional=original_notional,
            original_qty=original_qty,
            decided_at=datetime.now(UTC),
        )
    except Exception:
        logger.exception(
            "Failed to record HITL approval after fill (correlation_id=%s)",
            approved.correlation_id,
        )


def _route_hitl(hitl_status: str | None, pending_approved: ApprovedTrade) -> dict[str, Any]:
    """Map hitl_status to graph state after HITL resume.

    *pending_approved* is the ``ApprovedTrade`` built before the interrupt so
    that the same Trade object (same UUID + idempotency_key) is used whether
    the human approves immediately or after a delay.
    """
    if hitl_status == HITLStatus.REJECTED:
        return {"cycle_outcome": CycleOutcome.REJECTED}
    if hitl_status == HITLStatus.EXPIRED:
        return {"cycle_outcome": CycleOutcome.REJECTED_TIMEOUT}
    if hitl_status == HITLStatus.APPROVED:
        return {"approved_trade": pending_approved.model_dump(mode="json")}
    return {
        "cycle_outcome": CycleOutcome.ERROR,
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
    """Return an ``execution_node`` closed over injected ports.

    The fill is gated by NYSE market hours: if ``decision_ts`` falls outside
    regular trading hours (weekend, holiday, before-open, after-close,
    half-day past early close) the node returns
    ``CycleOutcome.REJECTED_MARKET_CLOSED`` without writing to the ledger.
    Research and decision nodes are unaffected — only the fill is gated.
    """
    agent = ExecutionAgent(ledger=ports.ledger, guardrail=ports.guardrail)

    def execution_node(state: GraphState) -> dict[str, Any]:
        from firm.agents.execution import Fill

        decision_ts_str = state.get("decision_ts", "")
        decision_ts = _parse_datetime(decision_ts_str)
        if not ports.calendar.is_market_open(decision_ts):
            logger.info(
                "Execution blocked: market closed at %s (correlation_id=%s)",
                decision_ts.isoformat(),
                state.get("correlation_id", ""),
            )
            return {"cycle_outcome": CycleOutcome.REJECTED_MARKET_CLOSED}

        approved_raw = state.get("approved_trade")
        if approved_raw is None:
            return {
                "cycle_outcome": CycleOutcome.ERROR,
                "error": "approved_trade missing in execution_node",
            }

        correlation_id = state.get("correlation_id", "")
        approved = _deserialise_approved_trade(approved_raw)
        if approved is None:
            return {
                "cycle_outcome": CycleOutcome.ERROR,
                "error": "approved_trade could not be deserialised",
            }

        trade = approved.trade
        prices = {trade.symbol: trade.requested_price}
        # hitl_status is set when a human operator explicitly approved the trade
        # via the HITL interrupt path.  In that case the guardrail uses the HITL-
        # aware enforcement that accepts oversized proposals the human cleared.
        hitl_approved = state.get("hitl_status") == HITLStatus.APPROVED
        inp = ExecutionInput(
            approved_trade=approved,
            portfolio_id=ports.portfolio_id,
            portfolio=ports.portfolio,
            prices=prices,
            correlation_id=correlation_id,
            hitl_approved=hitl_approved,
        )
        result = agent.run(inp)
        if isinstance(result, Fill):
            if hitl_approved:
                _record_approval_after_fill_from_state(ports, approved, state)
            return {"cycle_outcome": CycleOutcome.FILLED}
        return {"cycle_outcome": CycleOutcome.ERROR, "error": result.reason}

    return execution_node


# ---------------------------------------------------------------------------
# Reporting node
# ---------------------------------------------------------------------------


def make_reporting_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``reporting_node`` closed over injected ports.

    Price collection:
        For each symbol held in the portfolio, the current bar's close price is
        fetched from ``ports.market_data`` at ``decision_ts``.  If a bar is
        unavailable (e.g. non-trading day), avg_cost is used as a fallback
        inside the agent — the node does not gate on price availability.

        SPY prices are fetched for the benchmark calculation:
          - ``prices["SPY"]``      — report-date close
          - ``prices["SPY_PREV"]`` — previous calendar-day close (the last bar
            before the report date's midnight boundary); used only when present.
    """
    agent = ReportingAgent(report_sink=ports.report_sink, ledger=ports.ledger)

    def reporting_node(state: GraphState) -> dict[str, Any]:
        from firm.agents.reporting import ReportFailure

        correlation_id = state.get("correlation_id", "")
        decision_ts_str = state.get("decision_ts", "")
        decision_ts = _parse_datetime(decision_ts_str)
        report_date = decision_ts.date()
        cycle_id = str_to_uuid(correlation_id)

        prices = _fetch_report_prices(ports, decision_ts)

        inp = ReportingInput(
            cycle_id=cycle_id,
            portfolio_id=ports.portfolio_id,
            report_date=report_date,
            correlation_id=correlation_id,
            prices=prices,
        )
        result = agent.run(inp)
        outcome = state.get("cycle_outcome", CycleOutcome.FILLED)

        _persist_cycle(ports, state, decision_ts, outcome)

        if isinstance(result, ReportFailure):
            return {"cycle_outcome": outcome}  # degrade gracefully; don't overwrite outcome
        return {"cycle_outcome": outcome}

    return reporting_node


def _fetch_report_prices(ports: NodePorts, decision_ts: datetime) -> dict[str, Decimal]:
    """Fetch current-bar close prices for all held symbols plus SPY benchmark.

    Never raises — missing bars are silently omitted so the agent can fall back
    to avg_cost for any symbol whose bar is unavailable.
    """
    from datetime import timedelta

    prices: dict[str, Decimal] = {}

    # Holdings
    for symbol in ports.portfolio.holdings:
        bar = ports.market_data.get_bar(symbol, decision_ts)
        if bar is not None:
            prices[symbol] = bar.close

    # SPY today (report date)
    spy_bar = ports.market_data.get_bar("SPY", decision_ts)
    if spy_bar is not None:
        prices["SPY"] = spy_bar.close

    # SPY previous trading day (for benchmark return calculation)
    prev_ts = decision_ts - timedelta(days=1)
    spy_prev_bar = ports.market_data.get_bar("SPY", prev_ts)
    if spy_prev_bar is not None:
        prices["SPY_PREV"] = spy_prev_bar.close

    return prices


# ---------------------------------------------------------------------------
# Debate nodes (bull → bear loop → research manager)
# ---------------------------------------------------------------------------

MAX_DEBATE_ROUNDS = 1  # one full bull+bear exchange by default


def make_bull_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    agent = DebaterAgent(llm=ports.llm, stance="bull")

    def bull_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        correlation_id = state.get("correlation_id", "")
        rounds = state.get("debate_rounds", 0)
        inp = DebaterInput(
            symbol=symbol,
            round_num=rounds + 1,
            correlation_id=correlation_id,
            stance="bull",
            evidence_summary=_evidence_text(state.get("evidence")),
            technical_summary=_technical_text(state.get("technical_signal")),
            opponent_history=list(state.get("bear_history") or []),
        )
        result = agent.run(inp)
        argument = (
            f"[Bull unavailable: {result.failure_reason}]"
            if isinstance(result, DebaterFailure)
            else result.argument
        )
        existing: list[str] = list(state.get("bull_history") or [])
        return {"bull_history": [*existing, argument]}

    return bull_node


def make_bear_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    agent = DebaterAgent(llm=ports.llm, stance="bear")

    def bear_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        correlation_id = state.get("correlation_id", "")
        rounds = state.get("debate_rounds", 0)
        inp = DebaterInput(
            symbol=symbol,
            round_num=rounds + 1,
            correlation_id=correlation_id,
            stance="bear",
            evidence_summary=_evidence_text(state.get("evidence")),
            technical_summary=_technical_text(state.get("technical_signal")),
            opponent_history=list(state.get("bull_history") or []),
        )
        result = agent.run(inp)
        argument = (
            f"[Bear unavailable: {result.failure_reason}]"
            if isinstance(result, DebaterFailure)
            else result.argument
        )
        existing: list[str] = list(state.get("bear_history") or [])
        return {
            "bear_history": [*existing, argument],
            "debate_rounds": rounds + 1,
        }

    return bear_node


def make_research_manager_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    agent = ResearchManagerAgent(llm=ports.llm)

    def research_manager_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        correlation_id = state.get("correlation_id", "")

        # Demo/override: when force_buy is set, skip the LLM call and inject a
        # synthetic high-conviction BUY plan so the pm_node sizes a large enough
        # position to cross the HITL threshold (conviction=1.0 → 10% NAV).
        if state.get("force_buy"):
            return {"research_plan": _forced_buy_plan(symbol, correlation_id)}

        inp = ResearchManagerInput(
            symbol=symbol,
            correlation_id=correlation_id,
            evidence_summary=_evidence_text(state.get("evidence")),
            technical_summary=_technical_text(state.get("technical_signal")),
            bull_history=list(state.get("bull_history") or []),
            bear_history=list(state.get("bear_history") or []),
        )
        result = agent.run(inp)
        if isinstance(result, ResearchManagerFailure):
            return {
                "research_plan": {
                    "failure_reason": result.failure_reason,
                    "symbol": result.symbol,
                    "correlation_id": result.correlation_id,
                }
            }
        return {"research_plan": result.model_dump(mode="json")}

    return research_manager_node


def _forced_buy_plan(symbol: str, correlation_id: str) -> dict[str, Any]:
    """Return a synthetic ResearchPlan dict for the HITL demo/override path.

    Conviction 1.0 with max_trade_notional_pct=0.10 produces a 10% NAV trade
    (~$10 000 on a $100 000 portfolio) — well above the 5% HITL threshold.
    This is clearly named as an override and does NOT affect the default path.
    """
    return {
        "symbol": symbol,
        "correlation_id": correlation_id,
        "recommendation": "strong_buy",
        "conviction": 1.0,
        "bull_summary": "DEMO: forced BUY override for HITL end-to-end testing.",
        "bear_summary": "DEMO: no bear case — this is a synthetic override.",
        "rationale": "DEMO/FORCE-BUY override — synthetic plan for HITL end-to-end testing",
    }


# ---------------------------------------------------------------------------
# Technical analysis node
# ---------------------------------------------------------------------------


def make_technical_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``technical_node`` closed over injected ports.

    Runs in parallel with ``research_node`` (both fan out from START).
    PM waits for both before making a proposal.
    """
    agent = TechnicalAnalysisAgent(market_data=ports.market_data, llm=ports.llm)

    def technical_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        correlation_id = state.get("correlation_id", "")
        decision_ts = _parse_datetime(decision_ts_str)
        inp = TechnicalInput(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
        )
        result = agent.run(inp)
        return {"technical_signal": result.model_dump(mode="json")}

    return technical_node


# ---------------------------------------------------------------------------
# Synthesis node
# ---------------------------------------------------------------------------


def make_synthesis_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``synthesis_node`` that writes an LLM-authored investment memo."""
    agent = SynthesisReportAgent(llm=ports.llm)

    def synthesis_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        correlation_id = state.get("correlation_id", "")
        decision_ts = _parse_datetime(decision_ts_str)
        inp = SynthesisInput(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
            evidence=state.get("evidence"),
            technical_signal=state.get("technical_signal"),
            research_plan=state.get("research_plan"),
            trade_proposal=state.get("trade_proposal"),
            cycle_outcome=state.get("cycle_outcome"),
        )
        result = agent.run(inp)
        return {"synthesis": result.model_dump(mode="json")}

    return synthesis_node


# ---------------------------------------------------------------------------
# Judge node (LLM-as-a-judge)
# ---------------------------------------------------------------------------


def make_judge_node(ports: NodePorts) -> Callable[[GraphState], dict[str, Any]]:
    """Return a ``judge_node`` that scores decision-cycle coherence."""
    agent = JudgeAgent(llm=ports.llm)

    def judge_node(state: GraphState) -> dict[str, Any]:
        symbol = state.get("symbol", "")
        decision_ts_str = state.get("decision_ts", "")
        correlation_id = state.get("correlation_id", "")
        decision_ts = _parse_datetime(decision_ts_str)
        inp = JudgeInput(
            symbol=symbol,
            decision_ts=decision_ts,
            correlation_id=correlation_id,
            evidence=state.get("evidence"),
            technical_signal=state.get("technical_signal"),
            research_plan=state.get("research_plan"),
            trade_proposal=state.get("trade_proposal"),
            cycle_outcome=state.get("cycle_outcome"),
            synthesis=state.get("synthesis"),
        )
        result = agent.run(inp)
        if isinstance(result, JudgeFailure):
            return {
                "verdict": {
                    "failure_reason": result.failure_reason,
                    "correlation_id": correlation_id,
                }
            }
        return {"verdict": result.model_dump(mode="json")}

    return judge_node


# ---------------------------------------------------------------------------
# Cycle persistence helpers
# ---------------------------------------------------------------------------


def _persist_cycle(
    ports: NodePorts,
    state: GraphState,
    decision_ts: datetime,
    outcome: str | None,
) -> None:
    """Build a CycleAuditRecord from GraphState and write it via the ledger.

    Failure-isolated: logs the exception and returns without aborting the cycle.
    Called by reporting_node for every outcome (hold/rejected/filled/error/…).
    """
    if ports.ledger is None:
        return
    try:
        record = _build_cycle_audit_record(state, decision_ts, outcome)
        ports.ledger.record_cycle(record)
    except Exception:
        logger.exception(
            "Failed to persist decision cycle (correlation_id=%s, outcome=%r)",
            state.get("correlation_id", ""),
            outcome,
        )


def _build_cycle_audit_record(
    state: GraphState,
    decision_ts: datetime,
    outcome: str | None,
) -> CycleAuditRecord:
    """Extract CycleAuditRecord fields from GraphState.

    Reads research_plan for recommendation/conviction, verdict for judge_score
    and alignment, and approved_trade for trade_id when the outcome is filled.
    """
    correlation_id = state.get("correlation_id", "")
    symbol = state.get("symbol", "")
    trigger_type = state.get("trigger_type", "scheduled")

    recommendation, conviction = _extract_research_plan_fields(state.get("research_plan"))
    judge_score, alignment = _extract_verdict_fields(state.get("verdict"))
    trade_id = _extract_trade_id(state.get("approved_trade"), outcome)

    return CycleAuditRecord(
        correlation_id=correlation_id,
        symbol=symbol,
        trigger_type=trigger_type,
        decision_ts=decision_ts,
        recommendation=recommendation,
        conviction=conviction,
        outcome=outcome or CycleOutcome.ERROR,
        judge_score=judge_score,
        alignment=alignment,
        trade_id=trade_id,
    )


def _extract_research_plan_fields(
    research_plan: dict[str, Any] | None,
) -> tuple[str | None, float | None]:
    """Return (recommendation, conviction) from a serialised research_plan dict."""
    if not research_plan:
        return None, None
    recommendation = research_plan.get("recommendation")
    conviction_raw = research_plan.get("conviction")
    conviction = float(conviction_raw) if conviction_raw is not None else None
    return recommendation, conviction


def _extract_verdict_fields(
    verdict: dict[str, Any] | None,
) -> tuple[int | None, str | None]:
    """Return (judge_score, alignment) from a serialised verdict dict."""
    if not verdict:
        return None, None
    score_raw = verdict.get("coherence_score")
    judge_score = int(score_raw) if score_raw is not None else None
    alignment = verdict.get("alignment")
    return judge_score, alignment


def _extract_trade_id(
    approved_trade: dict[str, Any] | None,
    outcome: str | None,
) -> uuid.UUID | None:
    """Return the trade UUID when the cycle produced a fill, else None."""
    if outcome != CycleOutcome.FILLED or approved_trade is None:
        return None
    trade_raw = approved_trade.get("trade", {})
    trade_id_str = trade_raw.get("id")
    if not trade_id_str:
        return None
    try:
        return uuid.UUID(str(trade_id_str))
    except (ValueError, AttributeError):
        return None


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


def _evidence_text(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "No fundamental evidence available."
    claims = evidence.get("claims", [])
    if not claims:
        return "Research found no usable claims."
    return "; ".join(c.get("text", "") for c in claims[:5])


def _technical_text(technical: dict[str, Any] | None) -> str:
    if not technical or "reason" in technical:
        return "No technical analysis available."
    bias = technical.get("bias", "neutral")
    rsi = technical.get("rsi", 0.0)
    headline = technical.get("headline", "")
    return f"Bias: {bias} | RSI: {rsi:.1f} | {headline}"


def _deserialise_research_plan(raw: dict[str, Any] | None) -> Any:
    from firm.agents.research_manager.schemas import ResearchPlan

    if raw is None:
        return None
    try:
        return ResearchPlan.model_validate(raw)
    except Exception:
        logger.exception("Failed to deserialise research_plan: %r", raw)
        return None


def _deserialise_technical_signal(raw: dict[str, Any] | None) -> Any:
    """Deserialise technical_signal dict to TechnicalSignal or TechnicalUnavailable."""
    from firm.agents.technical import TechnicalSignal, TechnicalUnavailable

    if raw is None:
        return None
    if "bias" in raw:
        try:
            return TechnicalSignal.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise TechnicalSignal: %r", raw)
    if "reason" in raw:
        try:
            return TechnicalUnavailable.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise TechnicalUnavailable: %r", raw)
    return None


def _deserialise_evidence(raw: dict[str, Any] | None) -> Any:
    """Deserialise evidence dict to Evidence or Refusal."""
    from firm.agents.research import Evidence, Refusal

    if raw is None:
        return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)
    if "claims" in raw:
        try:
            return Evidence.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise Evidence: %r", raw)
            return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)
    if "reason" in raw:
        try:
            return Refusal.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise Refusal: %r", raw)
            return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)
    return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)


def _deserialise_proposal(raw: dict[str, Any]) -> Any:
    """Deserialise proposal dict to TradeProposal or Hold."""
    if "qty" in raw and "notional" in raw:
        try:
            return TradeProposal.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise TradeProposal: %r", raw)
    if "reason" in raw:
        try:
            return Hold.model_validate(raw)
        except Exception:
            logger.exception("Failed to deserialise Hold: %r", raw)
    symbol = raw.get("symbol", "")
    return Hold(symbol=symbol, reason="deserialisation failed")


def _deserialise_approved_trade(raw: dict[str, Any]) -> ApprovedTrade | None:
    try:
        return ApprovedTrade.model_validate(raw)
    except Exception:
        logger.exception("Failed to deserialise ApprovedTrade: %r", raw)
        return None


def _extract_prices(proposal: object, portfolio: object) -> dict[str, Decimal]:
    """Extract a minimal prices dict for risk evaluation.

    Derives the proposal's implied price from notional/qty and uses avg_cost
    as a fallback for existing holdings when no live prices are available.
    """
    from firm.domain import Portfolio as PortfolioModel

    prices: dict[str, Decimal] = {}
    if isinstance(proposal, TradeProposal) and proposal.qty > Decimal("0"):
        prices[proposal.symbol] = proposal.notional / proposal.qty
    if isinstance(portfolio, PortfolioModel):
        for sym, holding in portfolio.holdings.items():
            if sym not in prices:
                prices[sym] = holding.avg_cost
    return prices
