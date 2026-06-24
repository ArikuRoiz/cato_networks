"""DebaterAgent — argues the bull or bear case; rebuts the opponent's last argument."""

from __future__ import annotations

from typing import Literal

from firm.agents.base import BaseAgent
from firm.agents.debater.schemas import DebaterCase, DebaterFailure, DebaterInput
from firm.domain.enums import LLMModel
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage
from firm.utils import parse_json_dict

_SYSTEM_PROMPTS: dict[str, str] = {
    "bull": (
        "You are a bull-side equity analyst at a quantitative trading firm. Build the "
        "strongest EVIDENCE-BASED case for taking or growing a position in this stock.\n"
        "Focus on: growth and catalysts (market opportunity, demand, upcoming events that "
        "could re-rate the stock); competitive edge (differentiated product, pricing power, "
        "market position); confirming signals (the fundamentals and technicals that support "
        "the upside); and rebuttal — engage the bear's LAST argument directly and show why "
        "it is overstated or already priced in, rather than listing your own points beside it.\n"
        "Argue ONLY from the evidence and technical signal provided; do not invent numbers, "
        "prices, dates, or facts you were not given. If fundamental evidence is thin, anchor "
        "the case on the technical signal — never decline.\n"
        "Respond ONLY with valid JSON, no markdown fences."
    ),
    "bear": (
        "You are a bear-side equity analyst at a quantitative trading firm. Build the "
        "strongest EVIDENCE-BASED case against opening or holding a position in this stock.\n"
        "Focus on: risks and challenges (market saturation, financial fragility, macro "
        "headwinds); competitive weakness (eroding moat, declining innovation, stronger "
        "rivals); warning signals (the fundamentals and technicals that point down); and "
        "rebuttal — engage the bull's LAST argument directly and expose its weak or "
        "over-optimistic assumptions, rather than listing your own points beside it.\n"
        "Argue ONLY from the evidence and technical signal provided; do not invent numbers, "
        "prices, dates, or facts you were not given. If fundamental evidence is thin, anchor "
        "the case on the technical signal — never decline.\n"
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
        "Identify catalysts, upside potential, and why risks are manageable. "
        "If fundamentals are thin, anchor the case on the technical signal above."
    ),
    "bear": (
        "Argue the strongest bear case. Be specific — cite evidence of risks, "
        "overvaluation, competition, macro headwinds, or technical weakness. "
        "Explain why the bull is wrong or overlooking key risks. "
        "If fundamentals are thin, anchor the case on the technical signal above."
    ),
}


class DebaterAgent(BaseAgent[DebaterInput, DebaterCase | DebaterFailure]):
    def __init__(self, llm: LLM, stance: Literal["bull", "bear"]) -> None:
        self._llm = llm
        self._stance = stance

    def run(self, inp: DebaterInput) -> DebaterCase | DebaterFailure:
        resp = self._llm.complete(_build_messages(inp), model=LLMModel.HAIKU, max_tokens=768)
        if isinstance(resp, LLMError):
            # A transient LLM hiccup must not silently kill the debate: fall back
            # to a case built from the available technicals/evidence so the
            # research manager still receives a substantive argument to weigh.
            return _fallback_case(inp)
        raw = parse_json_dict(resp.content)
        if raw is None:
            return _fallback_case(inp)
        argument = str(raw.get("argument", "")).strip()
        key_points = [str(p).strip() for p in raw.get("key_points", []) if str(p).strip()]
        if not argument and not key_points:
            # The model returned parseable JSON but no usable content — degrade to
            # the deterministic fallback rather than emit an empty case.
            return _fallback_case(inp)
        return DebaterCase(
            symbol=inp.symbol,
            round_num=inp.round_num,
            stance=inp.stance,
            argument=argument or _fallback_argument(inp),
            key_points=key_points or _fallback_points(inp),
        )


_FALLBACK_FRAMING: dict[str, str] = {
    "bull": (
        "The strongest available upside case rests on the technical signal and any "
        "fundamental evidence on hand"
    ),
    "bear": (
        "The strongest available downside case rests on the technical signal and any "
        "fundamental evidence on hand"
    ),
}


def _fallback_case(inp: DebaterInput) -> DebaterCase:
    """Build a usable case from the supplied context when the LLM fails.

    Used on an LLM error or an empty/unparseable completion so a transient
    hiccup or thin-evidence cycle still yields a substantive argument. Quotes
    only the signals already passed in — it fabricates no prices or facts.
    """
    return DebaterCase(
        symbol=inp.symbol,
        round_num=inp.round_num,
        stance=inp.stance,
        argument=_fallback_argument(inp),
        key_points=_fallback_points(inp),
    )


def _fallback_argument(inp: DebaterInput) -> str:
    return (
        f"{_FALLBACK_FRAMING[inp.stance]} for {inp.symbol}. "
        f"Technical read: {inp.technical_summary or 'no technical data'}. "
        f"Fundamental evidence: {inp.evidence_summary or 'none on hand'}. "
        f"On the {inp.stance} side, this signal warrants attention rather than dismissal."
    )


def _fallback_points(inp: DebaterInput) -> list[str]:
    points: list[str] = []
    if inp.technical_summary:
        points.append(f"Technical signal: {inp.technical_summary}")
    if inp.evidence_summary:
        points.append(f"Fundamental evidence: {inp.evidence_summary}")
    return points or [f"{inp.stance.capitalize()} case rests on the available signal."]


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
