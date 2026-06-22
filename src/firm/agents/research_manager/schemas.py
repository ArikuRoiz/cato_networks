"""Research manager agent I/O schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from firm.domain.enums import Recommendation


class ResearchManagerInput(BaseModel):
    symbol: str
    correlation_id: str
    evidence_summary: str = ""
    technical_summary: str = ""
    bull_history: list[str] = []
    bear_history: list[str] = []

    model_config = {"frozen": True}


class ResearchPlan(BaseModel):
    """Adjudicated output from the Research Manager.

    ``recommendation`` maps to a PM signal:
      strong_buy  -> +1.0 * conviction
      buy         -> +0.5 * conviction
      hold        ->  0.0
      sell        -> -0.5 * conviction
      strong_sell -> -1.0 * conviction
    """

    symbol: str
    correlation_id: str
    recommendation: Recommendation
    conviction: float = Field(ge=0.0, le=1.0)
    bull_summary: str
    bear_summary: str
    rationale: str

    model_config = {"frozen": True}

    @property
    def signal_score(self) -> float:
        weights = {
            Recommendation.STRONG_BUY: 1.0,
            Recommendation.BUY: 0.5,
            Recommendation.HOLD: 0.0,
            Recommendation.SELL: -0.5,
            Recommendation.STRONG_SELL: -1.0,
        }
        return weights[self.recommendation] * self.conviction


class ResearchManagerFailure(BaseModel):
    symbol: str
    correlation_id: str
    failure_reason: str

    model_config = {"frozen": True}
