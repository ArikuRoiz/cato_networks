"""PolicyResult union — the three outcomes of a risk check."""

from __future__ import annotations

from pydantic import BaseModel


class Approved(BaseModel):
    """Trade passed all RiskPolicy checks."""

    model_config = {"frozen": True}


class HITLRequired(BaseModel):
    """Trade requires human-in-the-loop approval before execution."""

    reason: str

    model_config = {"frozen": True}


class Rejected(BaseModel):
    """Trade was hard-rejected by RiskPolicy."""

    reason: str

    model_config = {"frozen": True}


PolicyResult = Approved | HITLRequired | Rejected
