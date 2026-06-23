"""Portfolio manager schemas — TradeProposal and Hold.

The ``PortfolioManagerAgent`` class has been dissolved into the deterministic
``size_position`` tool (``firm.tools.size_position``).  Only the output schemas
are kept here because ``RiskAgent``, ``ExecutionAgent``, and the eval harness
all import ``TradeProposal`` / ``Hold`` from this package.
"""

from firm.agents.portfolio_manager.schemas import Hold, TradeProposal

__all__ = ["Hold", "TradeProposal"]
