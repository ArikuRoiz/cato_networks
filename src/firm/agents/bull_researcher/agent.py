"""BullResearcherAgent — argues the strongest upside case for a stock."""

from __future__ import annotations

from firm.agents.base import BaseAgent
from firm.agents.bull_researcher.schemas import BullCase, BullFailure, BullInput
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage
from firm.utils import parse_json_dict

_SYSTEM_PROMPT = (
    "You are a senior equity analyst specialising in identifying bullish investment theses. "
    "Your job is to argue the strongest possible upside case for a stock based on the "
    "available evidence. If a bear argument exists, rebut it specifically. "
    "Respond ONLY with valid JSON, no markdown fences."
)

_JSON_SCHEMA = (
    '{"argument":"<2-3 paragraph bull case>","key_points":["<specific bullish factor>",...]}'
)


class BullResearcherAgent(BaseAgent[BullInput, BullCase | BullFailure]):
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, inp: BullInput) -> BullCase | BullFailure:
        resp = self._llm.complete(_build_messages(inp), model="haiku", max_tokens=768)
        if isinstance(resp, LLMError):
            return BullFailure(symbol=inp.symbol, round_num=inp.round_num, failure_reason=f"llm_error: {resp.message}")
        raw = parse_json_dict(resp.content)
        if raw is None:
            return BullFailure(symbol=inp.symbol, round_num=inp.round_num, failure_reason="non-object JSON")
        return BullCase(
            symbol=inp.symbol,
            round_num=inp.round_num,
            argument=str(raw.get("argument", "")),
            key_points=[str(p) for p in raw.get("key_points", [])],
        )


def _build_messages(inp: BullInput) -> list[LLMMessage]:
    bear_section = f"\n\nBEAR'S PREVIOUS ARGUMENT (rebut this):\n{inp.bear_history[-1]}" if inp.bear_history else ""
    user = (
        f"Symbol: {inp.symbol} — Round {inp.round_num} (BULL)\n\n"
        f"FUNDAMENTAL EVIDENCE:\n{inp.evidence_summary or 'No evidence available.'}\n\n"
        f"TECHNICAL ANALYSIS:\n{inp.technical_summary or 'No technical data.'}"
        f"{bear_section}\n\n"
        f"Argue the strongest bull case. Be specific — cite the evidence and technicals. "
        f"Identify catalysts, upside potential, and why risks are manageable.\n\n"
        f"Respond ONLY with:\n{_JSON_SCHEMA}"
    )
    return [LLMMessage(role="system", content=_SYSTEM_PROMPT), LLMMessage(role="user", content=user)]
