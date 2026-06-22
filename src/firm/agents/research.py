"""ResearchAgent — retrieve evidence chunks, injection-scan, then synthesise claims.

Input:  ResearchInput(symbol, decision_ts, correlation_id)
Output: Evidence | Refusal

The agent NEVER emits prices, quantities, P&L, or dates — those come from
domain tools.  The LLM summarises only the retrieved text chunks.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from firm.domain.guardrails import InjectionDetected, InjectionGuard
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.types import Chunk, LLMError, LLMMessage

# ---------------------------------------------------------------------------
# I/O schemas
# ---------------------------------------------------------------------------


class ResearchInput(BaseModel):
    """Input contract for ResearchAgent."""

    symbol: str
    decision_ts: datetime
    correlation_id: str

    model_config = {"frozen": True}


class Claim(BaseModel):
    """A single grounded claim extracted from retrieved text."""

    text: str
    source_url: str
    chunk_id: str

    model_config = {"frozen": True}


class Evidence(BaseModel):
    """Successful research output: grounded claims for one symbol."""

    symbol: str
    claims: list[Claim]
    retrieved_at: datetime

    model_config = {"frozen": True}


class Refusal(BaseModel):
    """Research could not proceed — failure is a value, not an exception."""

    reason: Literal[
        "insufficient_evidence",
        "store_unavailable",
        "injection_detected",
    ]

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a financial research assistant. "
    "Summarise the key facts from the provided news excerpts. "
    "Respond ONLY with a JSON array of objects, each with keys "
    '"text" (string, ≤120 chars) and "chunk_id" (string). '
    "Do NOT emit prices, quantities, P&L figures, or dates. "
    "Use ONLY information present in the excerpts."
)


class ResearchAgent:
    """Retrieve, scan, and synthesise evidence for a single symbol."""

    def __init__(
        self,
        evidence: EvidenceStore,
        llm: LLM,
        injection_guard: InjectionGuard,
    ) -> None:
        self._evidence = evidence
        self._llm = llm
        self._injection_guard = injection_guard

    def run(self, inp: ResearchInput) -> Evidence | Refusal:
        """Return Evidence on success or a Refusal describing the failure mode."""
        chunks = self._evidence.search(
            inp.symbol,
            before=inp.decision_ts,
            k=10,
            query=f"{inp.symbol} earnings revenue guidance",
        )
        if not chunks:
            return Refusal(reason="insufficient_evidence")

        safe_chunks = _filter_safe_chunks(chunks, self._injection_guard)
        if not safe_chunks:
            return Refusal(reason="injection_detected")

        messages = _build_messages(inp.symbol, safe_chunks)
        resp = self._llm.complete(messages, model="haiku", max_tokens=1024)
        if isinstance(resp, LLMError):
            reason = "llm_error_non_retryable" if not resp.retryable else "llm_error_retryable"
            return Refusal(reason=reason)

        claims = _parse_claims(resp.content, safe_chunks)
        return Evidence(
            symbol=inp.symbol,
            claims=claims,
            retrieved_at=inp.decision_ts,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_safe_chunks(
    chunks: list[Chunk],
    guard: InjectionGuard,
) -> list[Chunk]:
    """Return only chunks that pass the injection scan."""
    return [c for c in chunks if not isinstance(guard.scan(c.text), InjectionDetected)]


def _build_messages(symbol: str, chunks: list[Chunk]) -> list[LLMMessage]:
    """Build the system + user message list for claim extraction."""
    excerpts = "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.text}" for c in chunks)
    user_content = (
        f"Symbol: {symbol}\n\nNews excerpts:\n{excerpts}\n\nExtract key claims as a JSON array."
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


def _parse_claims(content: str, chunks: list[Chunk]) -> list[Claim]:
    """Parse LLM output into a list of Claim objects with source attribution.

    Falls back to an empty list rather than raising on malformed output.
    """
    chunk_map = {c.chunk_id: c.source_url for c in chunks}
    try:
        raw = json.loads(content)
        if not isinstance(raw, list):
            return []
        return [
            Claim(
                text=item["text"],
                source_url=chunk_map.get(item["chunk_id"], ""),
                chunk_id=item["chunk_id"],
            )
            for item in raw
            if isinstance(item, dict) and "text" in item and "chunk_id" in item
        ]
    except (json.JSONDecodeError, KeyError):
        return []
