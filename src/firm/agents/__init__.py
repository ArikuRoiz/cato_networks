"""Typed agents for the AI Investment Firm pipeline.

The ``PortfolioManagerAgent`` has been dissolved — sizing is now handled by
the deterministic ``size_position`` tool in ``firm.tools.size_position``.
``TradeProposal`` and ``Hold`` schemas are retained for use by RiskAgent,
ExecutionAgent, and the eval harness.
"""

from firm.agents.execution import ExecutionAgent, ExecutionFailure, ExecutionInput, Fill
from firm.agents.portfolio_manager.schemas import Hold, TradeProposal
from firm.agents.reporting import ReportFailure, ReportingAgent, ReportingInput, ReportSent
from firm.agents.research import Claim, Evidence, Refusal, ResearchAgent, ResearchInput
from firm.agents.risk import ApprovedTrade, HITLRequired, Rejected, RiskAgent, RiskInput

__all__ = [
    "ApprovedTrade",
    "Claim",
    "Evidence",
    "ExecutionAgent",
    "ExecutionFailure",
    "ExecutionInput",
    "Fill",
    "HITLRequired",
    "Hold",
    "Refusal",
    "Rejected",
    "ReportFailure",
    "ReportSent",
    "ReportingAgent",
    "ReportingInput",
    "ResearchAgent",
    "ResearchInput",
    "RiskAgent",
    "RiskInput",
    "TradeProposal",
]
