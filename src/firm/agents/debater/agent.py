"""DebaterAgent — argues the bull or bear case; rebuts the opponent's last argument."""

from __future__ import annotations

from typing import Literal

from firm.agents.base import BaseAgent
from firm.agents.debater.schemas import DebaterCase, DebaterFailure, DebaterInput
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage
from firm.utils import parse_json_dict

_SYSTEM_PROMPTS: dict[str, str] = {
    "bull": (
        "You are a senior equity analyst specialising in identifying bullish investment theses. "
        "Your job is to argue the strongest possible upside case for a stock based on the "
        "available evidence. If a bear argument exists, rebut it specifically. "
        "Respond ONLY with valid JSON, no markdown fences."
    ),
    "bear": (
        "You are a senior equity analyst specialising in identifying bearish investment theses. "
        "Your job is to argue the strongest possible downside case for a stock based on the "
        "available evidence. Rebut the bull's argument specifically and identify overlooked risks. "
        "Respond ONLY with valid JSON, no markdown fences."
    ),
}

_JSON_SCHEMA: dict[str, str] = {
    "bull": '{"argument":"<2-3 paragraph bull case>","key_points":["<specific bullish factor>",...]}',
    "bear": '{"argument":"<2-3 paragraph bear case>","key_points":["<specific bearish risk>",...]}',
}

_OPPONENT_LABELS: dict[str, str] = {
    "bull": "BEAR'S PREVIOUS ARGUMENT (rebut this)",
    "bear": "BULL'S ARGUMENT (rebut this)",
}

_INSTRUCTION: dict[str, str] = {
    "bull": (
        "Argue the strongest bull case. Be specific — cite the evidence and technicals. "
        "Identify catalysts, upside potential, and why risks are manageable."
    ),
    "bear": (
        "Argue the strongest bear case. Be specific — cite evidence of risks, "
        "overvaluation, competition, macro headwinds, or technical weakness. "
        "Explain why the bull is wrong or overlooking key risks."
    ),
}


class DebaterAgent(BaseAgent[DebaterInput, DebaterCase | DebaterFailure]):
    def __init__(self, llm: LLM, stance: Literal["bull", "bear"]) -> None:
        self._llm = llm
        self._stance = stance

    def run(self, inp: DebaterInput) -> DebaterCase | DebaterFailure:
        resp = self._llm.complete(_build_messages(inp), model="haiku", max_tokens=768)
        if isinstance(resp, LLMError):
            return DebaterFailure(
                symbol=inp.symbol,
                round_num=inp.round_num,
                stance=inp.stance,
                failure_reason=f"llm_error: {resp.message}",
            )
        raw = parse_json_dict(resp.content)
        if raw is None:
            return DebaterFailure(
                symbol=inp.symbol,
                round_num=inp.round_num,
                stance=inp.stance,
                failure_reason="non-object JSON",
            )
        return DebaterCase(
            symbol=inp.symbol,
            round_num=inp.round_num,
            stance=inp.stance,
            argument=str(raw.get("argument", "")),
            key_points=[str(p) for p in raw.get("key_points", [])],
        )


def _build_messages(inp: DebaterInput) -> list[LLMMessage]:
    opponent_section = (
        f"\n\n{_OPPONENT_LABELS[inp.stance]}:\n{inp.opponent_history[-1]}"
        if inp.opponent_history
        else ""
    )
    user = (
        f"Symbol: {inp.symbol} — Round {inp.round_num} ({inp.stance.upper()})\n\n"
        f"FUNDAMENTAL EVIDENCE:\n{inp.evidence_summary or 'No evidence available.'}\n\n"
        f"TECHNICAL ANALYSIS:\n{inp.technical_summary or 'No technical data.'}"
        f"{opponent_section}\n\n"
        f"{_INSTRUCTION[inp.stance]}\n\n"
        f"Respond ONLY with:\n{_JSON_SCHEMA[inp.stance]}"
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPTS[inp.stance]),
        LLMMessage(role="user", content=user),
    ]
