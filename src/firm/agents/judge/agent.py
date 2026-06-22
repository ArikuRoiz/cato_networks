"""JudgeAgent — LLM-as-a-judge that audits each decision cycle for coherence."""

from __future__ import annotations

import json

from firm.agents.base import BaseAgent
from firm.agents.judge.schemas import JudgeInput, Verdict
from firm.ports.llm import LLM, LLMError
from firm.ports.types import LLMMessage

_SYSTEM_PROMPT = (
    "You are an independent risk auditor at a quantitative trading firm. "
    "Your sole job is to find logical gaps and inconsistencies in trading decisions. "
    "Be specific and critical — generic observations are useless. "
    "Respond ONLY with valid JSON, no markdown fences, no extra text."
)

_JSON_SCHEMA = (
    '{"coherence_score":<int 1-5>,'
    '"alignment":"aligned"|"partial"|"misaligned",'
    '"flags":["<specific concern>",...],'
    '"recommendation":"<1-sentence actionable recommendation>",'
    '"reasoning":"<2-3 sentences explaining the score>"}'
)

_RUBRIC = """
Coherence scoring rubric:
5 — Evidence, TA, and PM decision are mutually reinforcing; position size appropriate; no red flags.
4 — Minor inconsistency or missing signal but overall decision is defensible.
3 — Moderate misalignment (e.g. bearish TA ignored, buy signal on weak evidence).
2 — Significant inconsistency; decision contradicts primary signals without explanation.
1 — Evidence and TA both negative but a buy was executed, or vice versa.
"""


class JudgeAgent(BaseAgent[JudgeInput, Verdict]):
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def run(self, inp: JudgeInput) -> Verdict:
        messages = _build_messages(inp)
        resp = self._llm.complete(messages, model="sonnet", max_tokens=512)

        if isinstance(resp, LLMError):
            return _fallback_verdict(inp.correlation_id, f"llm_error: {resp.message}")

        return _parse_verdict(inp.correlation_id, resp.content)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _evidence_line(evidence: dict | None) -> str:  # type: ignore[type-arg]
    if not evidence:
        return "None (research failed or returned refusal)"
    claims = evidence.get("claims", [])
    if not claims:
        return "Research returned 0 usable claims"
    texts = [c.get("text", "") for c in claims[:3]]
    return "; ".join(texts)


def _technical_line(technical: dict | None) -> str:  # type: ignore[type-arg]
    if not technical or "reason" in technical:
        return "Unavailable"
    bias = technical.get("bias", "neutral")
    rsi = technical.get("rsi", 0.0)
    cross = technical.get("macd_cross", "none")
    headline = technical.get("headline", "")
    return f"bias={bias}, RSI={rsi:.1f}, MACD_cross={cross}. {headline}"


def _proposal_line(proposal: dict | None) -> str:  # type: ignore[type-arg]
    if not proposal:
        return "No proposal"
    if "qty" in proposal:
        return (
            f"{proposal.get('side', '?').upper()} {proposal.get('qty', '?')} shares "
            f"(${proposal.get('notional', '?')}). {proposal.get('rationale', '')}"
        )
    return f"Hold — {proposal.get('reason', '')}"


def _synthesis_line(synthesis: dict | None) -> str:  # type: ignore[type-arg]
    if not synthesis or "reason" in synthesis:
        return "Not available"
    return synthesis.get("executive_summary", "")


def _build_messages(inp: JudgeInput) -> list[LLMMessage]:
    date_str = inp.decision_ts.strftime("%Y-%m-%d")
    user = (
        f"Audit trading decision for {inp.symbol} on {date_str}.\n\n"
        f"Evidence: {_evidence_line(inp.evidence)}\n"
        f"Technical: {_technical_line(inp.technical_signal)}\n"
        f"PM decision: {_proposal_line(inp.trade_proposal)}\n"
        f"Cycle outcome: {inp.cycle_outcome or 'unknown'}\n"
        f"Synthesis summary: {_synthesis_line(inp.synthesis)}\n\n"
        f"{_RUBRIC}\n"
        f"Evaluate for: (1) evidence-decision alignment, "
        f"(2) TA-fundamental agreement, "
        f"(3) position sizing appropriateness, "
        f"(4) any red flags that were ignored.\n\n"
        f"Respond ONLY with:\n{_JSON_SCHEMA}"
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_verdict(correlation_id: str, content: str) -> Verdict:
    try:
        raw = json.loads(content.strip())
    except json.JSONDecodeError:
        return _fallback_verdict(correlation_id, "llm returned invalid JSON")

    score = int(raw.get("coherence_score", 3))
    score = max(1, min(5, score))
    alignment = raw.get("alignment", "partial")
    if alignment not in ("aligned", "partial", "misaligned"):
        alignment = "partial"

    return Verdict(
        correlation_id=correlation_id,
        coherence_score=score,
        alignment=alignment,  # type: ignore[arg-type]
        flags=[str(f) for f in raw.get("flags", [])],
        recommendation=str(raw.get("recommendation", "")),
        reasoning=str(raw.get("reasoning", "")),
    )


def _fallback_verdict(correlation_id: str, reason: str) -> Verdict:
    return Verdict(
        correlation_id=correlation_id,
        coherence_score=3,
        alignment="partial",
        flags=[f"Judge unavailable: {reason}"],
        recommendation="Manual review required.",
        reasoning="Judge agent failed to produce a verdict.",
    )
