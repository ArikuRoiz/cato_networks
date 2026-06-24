"""DebaterAgent — argues the bull or bear case; rebuts the opponent's last argument.

The debater can run in two modes:

* **Single-shot** (no evidence store, or no ``decision_ts``): one LLM call over
  the shared evidence + technical summaries.  Used by tests and as the fallback.
* **Tool-using** (evidence store + ``decision_ts`` supplied): the debater calls
  ``search_news`` to gather *its own* side-specific evidence — a bull hunts for
  catalysts, a bear for risks — then argues a case grounded in cited chunks.
  This is adversarial *retrieval*, not just adversarial argument.
"""

from __future__ import annotations

from typing import Any, Literal

from firm.agents.base import BaseAgent
from firm.agents.debater.schemas import DebaterCase, DebaterFailure, DebaterInput
from firm.agents.research.schemas import Claim
from firm.domain.enums import LLMModel
from firm.domain.guardrails import InjectionDetected, InjectionGuard
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, ToolDef
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


# Each debater searches for evidence supporting ITS OWN side before arguing.
_SEARCH_TOOL = ToolDef(
    name="search_news",
    description=(
        "Search recent news excerpts about the stock. Call it 1-2 times with queries "
        "that surface evidence for your side of the debate, then argue from what you find."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'NVDA data-center demand guidance'",
            },
            "k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (1-10, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)

_TOOL_SEARCH_GUIDANCE: dict[str, str] = {
    "bull": (
        " You have a search_news tool: run 1-2 targeted searches for evidence supporting "
        "the upside (catalysts, demand, guidance raises, positive analyst actions) before "
        "you argue. Every fact you cite must come from a retrieved excerpt, tagged by its "
        "chunk_id. Do not fabricate."
    ),
    "bear": (
        " You have a search_news tool: run 1-2 targeted searches for evidence supporting "
        "the downside (risks, competition, guidance cuts, negative analyst actions) before "
        "you argue. Every fact you cite must come from a retrieved excerpt, tagged by its "
        "chunk_id. Do not fabricate."
    ),
}

_TOOL_JSON_SCHEMA: dict[str, str] = {
    "bull": (
        '{"argument":"<2-3 paragraph bull case>","key_points":["<specific bullish factor>",...],'
        '"claims":[{"text":"<cited fact ≤120 chars>","chunk_id":"<id from an excerpt>"}]}'
    ),
    "bear": (
        '{"argument":"<2-3 paragraph bear case>","key_points":["<specific bearish risk>",...],'
        '"claims":[{"text":"<cited risk ≤120 chars>","chunk_id":"<id from an excerpt>"}]}'
    ),
}


class DebaterAgent(BaseAgent[DebaterInput, DebaterCase | DebaterFailure]):
    def __init__(
        self,
        llm: LLM,
        stance: Literal["bull", "bear"],
        evidence: EvidenceStore | None = None,
        injection_guard: InjectionGuard | None = None,
    ) -> None:
        self._llm = llm
        self._stance = stance
        self._evidence = evidence
        self._injection_guard = injection_guard

    def run(self, inp: DebaterInput) -> DebaterCase | DebaterFailure:
        if self._evidence is None or self._injection_guard is None or inp.decision_ts is None:
            return self._argue(inp)
        return self._research_then_argue(inp, self._evidence, self._injection_guard)

    def _argue(self, inp: DebaterInput) -> DebaterCase | DebaterFailure:
        """Single-shot path: argue over the shared evidence + technical summaries."""
        resp = self._llm.complete(_build_messages(inp), model=LLMModel.HAIKU, max_tokens=768)
        if isinstance(resp, LLMError):
            # A transient LLM hiccup must not silently kill the debate: fall back
            # to a case built from the available technicals/evidence so the
            # research manager still receives a substantive argument to weigh.
            return _fallback_case(inp)
        return _case_from_json(inp, parse_json_dict(resp.content), claims=[])

    def _research_then_argue(
        self, inp: DebaterInput, evidence: EvidenceStore, guard: InjectionGuard
    ) -> DebaterCase | DebaterFailure:
        """Tool path: gather side-specific cited evidence, then argue from it."""
        chunk_registry: dict[str, str] = {}  # chunk_id → source_url

        def search_news(args: dict[str, Any]) -> str:
            query = str(args.get("query", inp.symbol))
            k = min(int(args.get("k", 5)), 10)
            chunks = evidence.search(inp.symbol, before=inp.decision_ts, k=k, query=query)
            safe = [c for c in chunks if not isinstance(guard.scan(c.text), InjectionDetected)]
            if not safe:
                return "No safe news excerpts found for this query."
            for c in safe:
                chunk_registry[c.chunk_id] = c.source_url
            return "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.text[:500]}" for c in safe)

        resp = self._llm.complete_with_tools(
            _build_tool_messages(inp),
            tools=[_SEARCH_TOOL],
            executors={"search_news": search_news},
            model=LLMModel.HAIKU,
            max_tokens=900,
            max_rounds=3,  # ≤2 searches + final answer — bounded for token cost
        )
        if isinstance(resp, LLMError):
            return _fallback_case(inp)
        raw = parse_json_dict(resp.content)
        claims = _parse_debater_claims(raw, chunk_registry) if raw is not None else []
        return _case_from_json(inp, raw, claims=claims)


def _case_from_json(
    inp: DebaterInput, raw: dict[str, object] | None, *, claims: list[Claim]
) -> DebaterCase:
    """Build a DebaterCase from a parsed JSON object, degrading to fallback content."""
    if raw is None:
        return _fallback_case(inp)
    argument = str(raw.get("argument", "")).strip()
    key_points = [str(p).strip() for p in (raw.get("key_points") or []) if str(p).strip()]
    if not argument and not key_points:
        # Parseable JSON but no usable content — degrade rather than emit an empty case.
        return _fallback_case(inp)
    return DebaterCase(
        symbol=inp.symbol,
        round_num=inp.round_num,
        stance=inp.stance,
        argument=argument or _fallback_argument(inp),
        key_points=key_points or _fallback_points(inp),
        claims=claims,
    )


def _parse_debater_claims(raw: dict[str, object], chunk_registry: dict[str, str]) -> list[Claim]:
    """Extract cited claims, keeping only those tagged with a retrieved chunk_id.

    Claims whose chunk_id was never registered (i.e. the chunk was blocked by
    the injection guard or never returned by the evidence store) are silently
    dropped — this is the primary defence against the LLM hallucinating or
    citing a poisoned chunk that the guard rejected.
    """
    items = raw.get("claims")
    if not isinstance(items, list):
        return []
    return [
        Claim(
            text=str(item["text"]),
            chunk_id=str(item["chunk_id"]),
            source_url=chunk_registry[str(item["chunk_id"])],
        )
        for item in items
        if isinstance(item, dict)
        and "text" in item
        and "chunk_id" in item
        and str(item["chunk_id"]) in chunk_registry
    ]


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


def _build_tool_messages(inp: DebaterInput) -> list[LLMMessage]:
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
        "STEP 1 — Run 1-2 search_news calls to gather evidence for your side.\n"
        "STEP 2 — Rebut your opponent's last argument using what you found.\n"
        "STEP 3 — Respond ONLY with:\n"
        f"{_TOOL_JSON_SCHEMA[inp.stance]}"
    )
    return [
        LLMMessage(
            role="system", content=_SYSTEM_PROMPTS[inp.stance] + _TOOL_SEARCH_GUIDANCE[inp.stance]
        ),
        LLMMessage(role="user", content=user),
    ]


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
