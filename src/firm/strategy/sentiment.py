"""Sentiment signal computation — LLM-backed, evidence-grounded.

Calls the LLM port to rate aggregate sentiment from a set of retrieved
evidence claims.  Numbers never come from the LLM; the score is derived
by parsing a structured JSON response and clamping it to [-1.0, 1.0].
"""

from __future__ import annotations

import json

from firm.agents.research import Evidence
from firm.domain.enums import LLMModel
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage


def _sentiment_system_prompt(symbol: str) -> str:
    """Build the system prompt with the symbol interpolated per spec."""
    return (
        f"Given these news excerpts about {symbol}, "
        "rate the overall sentiment from -1.0 (very negative) to 1.0 (very positive). "
        'Return ONLY a JSON object: {"sentiment": <float>}'
    )


def compute_sentiment(
    evidence: Evidence,
    llm: LLM,
    model: LLMModel | str = LLMModel.HAIKU,
) -> float:
    """Rate overall sentiment for *evidence* using the LLM.

    Builds a prompt from the evidence claims, calls the LLM, and parses
    the returned JSON ``{"sentiment": <float>}``.  The score is clamped to
    ``[-1.0, 1.0]``.

    Returns ``0.0`` (neutral) on any LLM error or JSON parse failure so
    the caller always gets a usable float — failure is silent by design.

    Parameters
    ----------
    evidence:
        Grounded evidence output from ``ResearchAgent``.
    llm:
        ``LLM`` port implementation (injected — never constructed here).
    model:
        Model alias forwarded to ``llm.complete``.  Defaults to ``LLMModel.HAIKU``
        for cost-efficient extraction.
    """
    messages = _build_messages(evidence)
    result = llm.complete(messages, model=model, max_tokens=100)
    match result:
        case LLMError():
            return 0.0
        case _:
            return _parse_sentiment(result.content)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_messages(evidence: Evidence) -> list[LLMMessage]:
    """Construct the system + user message list for sentiment rating."""
    claim_text = "\n".join(f"- {claim.text}" for claim in evidence.claims)
    return [
        LLMMessage(role="system", content=_sentiment_system_prompt(evidence.symbol)),
        LLMMessage(role="user", content=claim_text),
    ]


def _parse_sentiment(content: str) -> float:
    """Parse ``{"sentiment": <float>}`` from *content* and clamp to [-1, 1].

    Returns ``0.0`` on any parse failure.
    """
    try:
        parsed = json.loads(content)
        raw = float(parsed["sentiment"])
        return _clamp(raw, -1.0, 1.0)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
