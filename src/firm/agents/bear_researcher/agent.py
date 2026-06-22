"""BearResearcherAgent — argues the strongest downside case for a stock."""

from __future__ import annotations

from firm.agents.base import BaseAgent
from firm.agents.bear_researcher.schemas import BearCase, BearFailure, BearInput
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage
from firm.utils import parse_json_dict

_SYSTEM_PROMPT = (
    "You are a senior equity analyst specialising in identifying bearish investment theses. "
    "Your job is to argue the strongest possible downside case for a stock based on the "
    "available evidence. Rebut the bull's argument specifically and identify overlooked risks. "
    "Respond ONLY with valid JSON, no markdown fences."
)

_JSON_SCHEMA = (
    '{"argument":"<2-3 paragraph bear case>","key_points":["<specific bearish risk>",...]}'
)


class BearResearcherAgent(BaseAgent[BearInput, BearCase | BearFailure]):
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, inp: BearInput) -> BearCase | BearFailure:
        resp = self._llm.complete(_build_messages(inp), model="haiku", max_tokens=768)
        if isinstance(resp, LLMError):
            return BearFailure(symbol=inp.symbol, round_num=inp.round_num, failure_reason=f"llm_error: {resp.message}")
        raw = parse_json_dict(resp.content)
        if raw is None:
            return BearFailure(symbol=inp.symbol, round_num=inp.round_num, failure_reason="non-object JSON")
        return BearCase(
            symbol=inp.symbol,
            round_num=inp.round_num,
            argument=str(raw.get("argument", "")),
            key_points=[str(p) for p in raw.get("key_points", [])],
        )


def _build_messages(inp: BearInput) -> list[LLMMessage]:
    bull_section = f"\n\nBULL'S ARGUMENT (rebut this):\n{inp.bull_history[-1]}" if inp.bull_history else ""
    user = (
        f"Symbol: {inp.symbol} — Round {inp.round_num} (BEAR)\n\n"
        f"FUNDAMENTAL EVIDENCE:\n{inp.evidence_summary or 'No evidence available.'}\n\n"
        f"TECHNICAL ANALYSIS:\n{inp.technical_summary or 'No technical data.'}"
        f"{bull_section}\n\n"
        f"Argue the strongest bear case. Be specific — cite evidence of risks, "
        f"overvaluation, competition, macro headwinds, or technical weakness. "
        f"Explain why the bull is wrong or overlooking key risks.\n\n"
        f"Respond ONLY with:\n{_JSON_SCHEMA}"
    )
    return [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user)]
