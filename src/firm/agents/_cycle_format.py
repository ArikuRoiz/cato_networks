"""Shared cycle-context formatting helpers.

Used by ``SynthesisReportAgent`` (paragraph form) and ``JudgeAgent`` (line form)
to render the same cycle data into their respective prompts.  Having one module
prevents the three copies that previously lived in synthesis/agent.py,
judge/agent.py, and orchestration/nodes.py from drifting apart.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Paragraph helpers — full prose, used by SynthesisReportAgent
# ---------------------------------------------------------------------------


def evidence_summary(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "No fundamental evidence retrieved."
    claims = evidence.get("claims", [])
    if not claims:
        return "Research ran but found no usable claims."
    lines = [f"- {c.get('text', '')}" for c in claims[:5]]
    return "\n".join(lines)


def technical_summary(technical: dict[str, Any] | None) -> str:
    if not technical or "reason" in technical:
        return "Technical analysis unavailable."
    bias = technical.get("bias", "neutral")
    headline = technical.get("headline", "")
    rsi = technical.get("rsi", 0.0)
    macd_cross = technical.get("macd_cross", "none")
    return f"Bias: {bias} | RSI: {rsi:.1f} | MACD cross: {macd_cross}\n{headline}"


def proposal_summary(proposal: dict[str, Any] | None) -> str:
    if not proposal:
        return "Hold — no proposal generated."
    if "qty" in proposal:
        side = proposal.get("side", "?")
        qty = proposal.get("qty", "?")
        notional = proposal.get("notional", "?")
        rationale = proposal.get("rationale", "")
        return f"{side.upper()} {qty} shares (${notional}) — {rationale}"
    return f"Hold — {proposal.get('reason', 'unknown reason')}"


def research_plan_summary(plan: dict[str, Any] | None) -> str:
    if not plan or "failure_reason" in plan:
        return "Research plan unavailable."
    rec = plan.get("recommendation", "unknown")
    conviction = float(plan.get("conviction", 0.0))
    rationale = plan.get("rationale", "")
    return f"Recommendation: {rec} (conviction: {conviction:.0%}). {rationale}"


# ---------------------------------------------------------------------------
# Terse line helpers — compact single-line, used by JudgeAgent
# ---------------------------------------------------------------------------


def evidence_line(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "None (research failed or returned refusal)"
    claims = evidence.get("claims", [])
    if not claims:
        return "Research returned 0 usable claims"
    texts = [c.get("text", "") for c in claims[:3]]
    return "; ".join(texts)


def technical_line(technical: dict[str, Any] | None) -> str:
    if not technical or "reason" in technical:
        return "Unavailable"
    bias = technical.get("bias", "neutral")
    rsi = technical.get("rsi", 0.0)
    cross = technical.get("macd_cross", "none")
    headline = technical.get("headline", "")
    return f"bias={bias}, RSI={rsi:.1f}, MACD_cross={cross}. {headline}"


def proposal_line(proposal: dict[str, Any] | None) -> str:
    if not proposal:
        return "No proposal"
    if "qty" in proposal:
        return (
            f"{proposal.get('side', '?').upper()} {proposal.get('qty', '?')} shares "
            f"(${proposal.get('notional', '?')}). {proposal.get('rationale', '')}"
        )
    return f"Hold — {proposal.get('reason', '')}"


def synthesis_line(synthesis: dict[str, Any] | None) -> str:
    if not synthesis or "reason" in synthesis:
        return "Not available"
    return str(synthesis.get("executive_summary", ""))


def research_plan_line(plan: dict[str, Any] | None) -> str:
    if not plan or "failure_reason" in plan:
        return "Unavailable"
    rec = plan.get("recommendation", "unknown")
    conviction = float(plan.get("conviction", 0.0))
    rationale = plan.get("rationale", "")
    return f"recommendation={rec}, conviction={conviction:.0%}. {rationale}"
