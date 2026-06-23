"""SynthesisReportAgent — LLM-written investment memo for the full decision cycle."""

from __future__ import annotations

import json

from firm.agents._cycle_format import (
    evidence_summary,
    proposal_summary,
    research_plan_summary,
    technical_summary,
)
from firm.agents.base import BaseAgent
from firm.agents.synthesis.schemas import (
    SynthesisFailure,
    SynthesisInput,
    SynthesisReport,
)
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage

_SYSTEM_PROMPT = (
    "You are a senior portfolio manager writing an internal investment memo for a "
    "quantitative trading firm. Your writing is precise, professional, and concise. "
    "Respond ONLY with valid JSON, no markdown fences, no extra text."
)

_JSON_SCHEMA = (
    '{"title":"<memo title>",'
    '"executive_summary":"<1-2 sentences>",'
    '"evidence_synthesis":"<paragraph: what research and TA showed>",'
    '"decision_rationale":"<paragraph: why the PM made this call>",'
    '"execution_quality":"<paragraph: how well the cycle executed>"}'
)


class SynthesisReportAgent(BaseAgent[SynthesisInput, SynthesisReport | SynthesisFailure]):
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, inp: SynthesisInput) -> SynthesisReport | SynthesisFailure:
        messages = _build_messages(inp)
        resp = self._llm.complete(messages, model="sonnet", max_tokens=1024)

        if isinstance(resp, LLMError):
            return SynthesisFailure(reason=f"llm_error: {resp.message}")

        return _parse_report(inp, resp.content)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_messages(inp: SynthesisInput) -> list[LLMMessage]:
    date_str = inp.decision_ts.strftime("%Y-%m-%d")
    user = (
        f"Investment decision memo for {inp.symbol} — {date_str}\n\n"
        f"FUNDAMENTAL EVIDENCE (Research Agent):\n{evidence_summary(inp.evidence)}\n\n"
        f"TECHNICAL ANALYSIS:\n{technical_summary(inp.technical_signal)}\n\n"
        f"DEBATE OUTCOME (Research Manager):\n{research_plan_summary(inp.research_plan)}\n\n"
        f"PORTFOLIO MANAGER DECISION:\n{proposal_summary(inp.trade_proposal)}\n\n"
        f"CYCLE OUTCOME: {inp.cycle_outcome or 'unknown'}\n\n"
        f"Write a professional investment memo. Respond ONLY with:\n{_JSON_SCHEMA}"
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_report(inp: SynthesisInput, content: str) -> SynthesisReport | SynthesisFailure:
    try:
        raw = json.loads(content.strip())
    except json.JSONDecodeError:
        return SynthesisFailure(reason="llm returned invalid JSON")
    if not isinstance(raw, dict):
        return SynthesisFailure(reason="llm returned non-object JSON")

    return SynthesisReport(
        symbol=inp.symbol,
        correlation_id=inp.correlation_id,
        title=str(raw.get("title", f"{inp.symbol} Decision Memo")),
        executive_summary=str(raw.get("executive_summary", "")),
        evidence_synthesis=str(raw.get("evidence_synthesis", "")),
        decision_rationale=str(raw.get("decision_rationale", "")),
        execution_quality=str(raw.get("execution_quality", "")),
    )
