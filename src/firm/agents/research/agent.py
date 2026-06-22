"""ResearchAgent — retrieve evidence chunks, injection-scan, then synthesise claims."""

from __future__ import annotations

import json

from firm.agents.base import BaseAgent
from firm.agents.research.schemas import Claim, Evidence, Refusal, ResearchInput
from firm.domain.guardrails import InjectionDetected, InjectionGuard
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.types import Chunk, LLMError, LLMMessage

_SYSTEM_PROMPT = (
    "You are a financial research assistant. "
    "Summarise the key facts from the provided news excerpts. "
    "Respond ONLY with a JSON array of objects, each with keys "
    '"text" (string, ≤120 chars) and "chunk_id" (string). '
    "Do NOT emit prices, quantities, P&L figures, or dates. "
    "Use ONLY information present in the excerpts."
)


class ResearchAgent(BaseAgent[ResearchInput, Evidence | Refusal]):
    def __init__(self, evidence: EvidenceStore, llm: LLM, injection_guard: InjectionGuard) -> None:
        self._evidence = evidence
        self._llm = llm
        self._injection_guard = injection_guard

    def run(self, inp: ResearchInput) -> Evidence | Refusal:
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
            if resp.retryable:
                return Refusal(reason="llm_error_retryable")
            return Refusal(reason="llm_error_non_retryable")

        claims = _parse_claims(resp.content, safe_chunks)
        return Evidence(symbol=inp.symbol, claims=claims, retrieved_at=inp.decision_ts)


def _filter_safe_chunks(chunks: list[Chunk], guard: InjectionGuard) -> list[Chunk]:
    return [c for c in chunks if not isinstance(guard.scan(c.text), InjectionDetected)]


def _build_messages(symbol: str, chunks: list[Chunk]) -> list[LLMMessage]:
    excerpts = "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.text}" for c in chunks)
    user_content = (
        f"Symbol: {symbol}\n\nNews excerpts:\n{excerpts}\n\nExtract key claims as a JSON array."
    )
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


def _parse_claims(content: str, chunks: list[Chunk]) -> list[Claim]:
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
