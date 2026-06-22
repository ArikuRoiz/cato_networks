"""Domain services — stateless helpers that coordinate ports and domain logic.

Services are distinct from agents (which produce typed outputs for the
LangGraph pipeline) and from adapters (which own external IO).  A service
encapsulates domain-relevant behaviour — such as market-calendar gating —
that depends on an external library but carries no port boundary.
"""

from firm.services.calendar import NYSECalendar

__all__ = ["NYSECalendar"]
