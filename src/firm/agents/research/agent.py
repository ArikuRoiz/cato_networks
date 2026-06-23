"""ResearchAgent — tool-using agent that searches for evidence and synthesises claims.

The LLM controls its own information gathering: it calls ``search_news`` with
whatever queries it needs, decides when it has enough evidence, and produces a
structured JSON array of factual claims.  This replaces the prior single-shot
approach where the query was hardcoded and the LLM received a fixed chunk set.
"""

from __future__ import annotations

import json
from typing import Any

from firm.agents.base import BaseAgent
from firm.agents.research.schemas import Claim, Evidence, Refusal, ResearchInput
from firm.domain.enums import RefusalReason
from firm.domain.guardrails import InjectionDetected, InjectionGuard
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.types import LLMError, LLMMessage, ToolDef

_SYSTEM_PROMPT = (
    "You are a financial research assistant. "
    "Use the search_news tool to find relevant evidence — call it multiple times "
    "with different queries to cover earnings, guidance, analyst outlook, and risks. "
    "When you have enough evidence, respond ONLY with a JSON array of claim objects, "
    'each with keys "text" (string, ≤120 chars) and "chunk_id" (string). '
    "Do NOT emit prices, quantities, P&L figures, or dates. "
    "Use ONLY information present in the retrieved excerpts."
)

_SEARCH_TOOL = ToolDef(
    name="search_news",
    description="Search for recent news excerpts about a stock symbol. Call multiple times with different queries to cover different aspects (earnings, guidance, analyst ratings, risks, sector trends).",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'NVDA earnings revenue guidance Q3'",
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


class ResearchAgent(BaseAgent[ResearchInput, Evidence | Refusal]):
    def __init__(self, evidence: EvidenceStore, llm: LLM, injection_guard: InjectionGuard) -> None:
        self._evidence = evidence
        self._llm = llm
        self._injection_guard = injection_guard

    def run(self, inp: ResearchInput) -> Evidence | Refusal:
        chunk_registry: dict[str, str] = {}  # chunk_id → source_url

        def search_news(args: dict[str, Any]) -> str:
            query = str(args.get("query", inp.symbol))
            k = min(int(args.get("k", 5)), 10)
            chunks = self._evidence.search(inp.symbol, before=inp.decision_ts, k=k, query=query)
            safe = [
                c
                for c in chunks
                if not isinstance(self._injection_guard.scan(c.text), InjectionDetected)
            ]
            if not safe:
                return "No safe news excerpts found for this query."
            for c in safe:
                chunk_registry[c.chunk_id] = c.source_url
            return "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.text[:500]}" for c in safe)

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Research {inp.symbol} for a trading decision on {inp.decision_ts.date()}. "
                    "Search for evidence on multiple aspects, then output your JSON claim array."
                ),
            ),
        ]

        resp = self._llm.complete_with_tools(
            messages,
            tools=[_SEARCH_TOOL],
            executors={"search_news": search_news},
            model="haiku",
            max_tokens=1024,
            max_rounds=6,
        )

        if isinstance(resp, LLMError):
            reason = (
                RefusalReason.LLM_ERROR_RETRYABLE
                if resp.retryable
                else RefusalReason.LLM_ERROR_NON_RETRYABLE
            )
            return Refusal(reason=reason)

        if not chunk_registry:
            return Refusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE)

        claims = _parse_claims(resp.content, chunk_registry)
        return Evidence(symbol=inp.symbol, claims=claims, retrieved_at=inp.decision_ts)


def _parse_claims(content: str, chunk_registry: dict[str, str]) -> list[Claim]:
    try:
        raw = json.loads(content)
        if not isinstance(raw, list):
            return []
        return [
            Claim(
                text=item["text"],
                chunk_id=item["chunk_id"],
                source_url=chunk_registry.get(item["chunk_id"], ""),
            )
            for item in raw
            if isinstance(item, dict) and "text" in item and "chunk_id" in item
        ]
    except (json.JSONDecodeError, KeyError):
        return []
