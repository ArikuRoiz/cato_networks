"""ResearchManagerAgent — adjudicates the bull/bear debate into a structured plan."""

from __future__ import annotations

from firm.agents.base import BaseAgent
from firm.agents.research_manager.schemas import (
    ResearchManagerFailure,
    ResearchManagerInput,
    ResearchPlan,
)
from firm.domain.enums import Recommendation
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage
from firm.utils import parse_json_dict

_SYSTEM_PROMPT = (
    "You are a research manager at a quantitative trading firm. "
    "You have just observed a structured debate between a bull analyst and a bear analyst. "
    "Your job is to objectively weigh their arguments against the underlying evidence "
    "and produce a clear, actionable recommendation. "
    "Be decisive — 'hold' is valid but requires justification. "
    "Respond ONLY with valid JSON, no markdown fences."
)

_VALID_RECS = {r.value for r in Recommendation}

_JSON_SCHEMA = (
    '{"recommendation":"strong_buy"|"buy"|"hold"|"sell"|"strong_sell",'
    '"conviction":<float 0.0-1.0>,'
    '"bull_summary":"<1-2 sentences summarising the bull case>",'
    '"bear_summary":"<1-2 sentences summarising the bear case>",'
    '"rationale":"<2-3 sentences explaining the final call>"}'
)


class ResearchManagerAgent(BaseAgent[ResearchManagerInput, ResearchPlan | ResearchManagerFailure]):
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, inp: ResearchManagerInput) -> ResearchPlan | ResearchManagerFailure:
        messages = _build_messages(inp)
        resp = self._llm.complete(messages, model="sonnet", max_tokens=512)
        if isinstance(resp, LLMError):
            return ResearchManagerFailure(
                symbol=inp.symbol,
                correlation_id=inp.correlation_id,
                failure_reason=f"llm_error: {resp.message}",
            )
        return _parse(inp, resp.content)


def _format_debate(bull_history: list[str], bear_history: list[str]) -> str:
    rounds: list[str] = []
    for i, (bull, bear) in enumerate(zip(bull_history, bear_history, strict=False), start=1):
        rounds.append(f"--- Round {i} ---\nBULL:\n{bull}\n\nBEAR:\n{bear}")
    return "\n\n".join(rounds) if rounds else "No debate recorded."


def _build_messages(inp: ResearchManagerInput) -> list[LLMMessage]:
    debate = _format_debate(inp.bull_history, inp.bear_history)
    user = (
        f"Adjudicate the investment debate for {inp.symbol}.\n\n"
        f"UNDERLYING EVIDENCE:\n{inp.evidence_summary or 'None available.'}\n\n"
        f"TECHNICAL ANALYSIS:\n{inp.technical_summary or 'None available.'}\n\n"
        f"DEBATE TRANSCRIPT:\n{debate}\n\n"
        f"Weigh each side. Who made the stronger, more evidence-grounded argument? "
        f"Produce a recommendation that a portfolio manager can act on.\n\n"
        f"Respond ONLY with:\n{_JSON_SCHEMA}"
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    ]


def _parse(inp: ResearchManagerInput, content: str) -> ResearchPlan | ResearchManagerFailure:
    raw = parse_json_dict(content)
    if raw is None:
        return ResearchManagerFailure(
            symbol=inp.symbol, correlation_id=inp.correlation_id, failure_reason="non-object JSON"
        )
    try:
        rec_str = str(raw.get("recommendation", Recommendation.HOLD))
        rec = Recommendation(rec_str) if rec_str in _VALID_RECS else Recommendation.HOLD
        conviction = max(0.0, min(1.0, float(raw.get("conviction", 0.5))))
        return ResearchPlan(
            symbol=inp.symbol,
            correlation_id=inp.correlation_id,
            recommendation=rec,
            conviction=conviction,
            bull_summary=str(raw.get("bull_summary", "")),
            bear_summary=str(raw.get("bear_summary", "")),
            rationale=str(raw.get("rationale", "")),
        )
    except (KeyError, ValueError) as exc:
        return ResearchManagerFailure(
            symbol=inp.symbol, correlation_id=inp.correlation_id, failure_reason=f"parse_error: {exc}"
        )
