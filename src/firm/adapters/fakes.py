"""In-memory fakes implementing port interfaces — for use in unit tests only.

Fakes are fast, deterministic, and have zero external dependencies.
They satisfy the runtime_checkable Protocol contracts so tests can verify
structural compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from firm.domain import Bar
from firm.domain.enums import ApprovalStatus
from firm.ports.types import (
    ApprovalResult,
    Chunk,
    DailyReport,
    HITLRequest,
    LLMError,
    LLMMessage,
    LLMResponse,
    NewsDoc,
    ToolDef,
    ToolExecutors,
)

# ---------------------------------------------------------------------------
# FakeMarketData
# ---------------------------------------------------------------------------


@dataclass
class FakeMarketData:
    """In-memory ``MarketDataSource``.

    Seed bars via ``add_bar``; lookup is O(1) by ``(symbol, ts)`` key.
    """

    bars: dict[tuple[str, datetime], Bar] = field(default_factory=dict)

    def add_bar(self, bar: Bar) -> None:
        """Register *bar* so it can be retrieved by ``get_bar``."""
        self.bars[(bar.symbol, bar.ts)] = bar

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        """Return the bar for *(symbol, ts)* or ``None`` if absent."""
        return self.bars.get((symbol, ts))

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Return bars for *symbol* in the half-open interval ``[start, end)``."""
        return sorted(
            (bar for (sym, ts), bar in self.bars.items() if sym == symbol and start <= ts < end),
            key=lambda b: b.ts,
        )


# ---------------------------------------------------------------------------
# FakeEvidenceStore
# ---------------------------------------------------------------------------


@dataclass
class FakeEvidenceStore:
    """In-memory ``EvidenceStore``.

    Stores chunks directly (no embedding computation).
    Enforces the no-lookahead rule: ``search`` only returns chunks with
    ``published_at <= before``.
    """

    docs: list[Chunk] = field(default_factory=list)

    def search(
        self,
        symbol: str,
        *,
        before: datetime,
        k: int = 10,
        query: str | None = None,
    ) -> list[Chunk]:
        """Filter by *symbol* and ``published_at <= before``, return first *k*.

        ``query`` is accepted for protocol compatibility but ignored — the fake
        has no embedding engine.
        """
        matching = [
            chunk for chunk in self.docs if chunk.symbol == symbol and chunk.published_at <= before
        ]
        return matching[:k]

    def embed_and_store(self, doc: NewsDoc) -> None:
        """Store *doc* as a ``Chunk`` with a zero score (no real embedding)."""
        chunk = Chunk(
            id=uuid4(),
            symbol=doc.symbol,
            text=doc.text,
            source_url=doc.source_url,
            chunk_id=f"fake-{uuid4().hex[:8]}",
            published_at=doc.published_at,
            score=0.0,
        )
        self.docs.append(chunk)


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


@dataclass
class FakeLLM:
    """In-memory ``LLM`` that replays a pre-loaded list of responses in order.

    Raises ``IndexError`` when the response list is exhausted so tests fail
    loudly rather than silently returning stale data.
    """

    responses: list[LLMResponse] = field(default_factory=list)
    index: int = 0

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse | LLMError:
        """Return ``responses[index]`` and advance the cursor.

        Raises ``IndexError`` when no more pre-loaded responses are available.
        """
        if self.index >= len(self.responses):
            raise IndexError(
                f"FakeLLM has no response at index {self.index}; "
                f"only {len(self.responses)} response(s) were loaded."
            )
        response = self.responses[self.index]
        self.index += 1
        return response

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDef],
        executors: ToolExecutors,
        *,
        model: str,
        max_tokens: int,
        max_rounds: int = 5,
    ) -> LLMResponse | LLMError:
        """Skip tool execution and return the next queued response directly."""
        return self.complete(messages, model=model, max_tokens=max_tokens)

    def count_tokens(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
    ) -> int:
        """Rough estimate: total content length divided by 4."""
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4


# ---------------------------------------------------------------------------
# FakeReportSink
# ---------------------------------------------------------------------------


@dataclass
class FakeReportSink:
    """In-memory ``ReportSink`` that captures outbound calls for assertion.

    ``send_hitl_request`` auto-approves so agent tests can proceed to execution
    without a human in the loop.
    """

    daily_reports_sent: list[DailyReport] = field(default_factory=list)
    hitl_requests: list[HITLRequest] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)

    def send_daily_report(self, report: DailyReport) -> None:
        """Capture *report* for later assertion."""
        self.daily_reports_sent.append(report)

    def send_hitl_request(self, req: HITLRequest) -> ApprovalResult:
        """Capture *req* and return an auto-approved result."""
        self.hitl_requests.append(req)
        return ApprovalResult(status=ApprovalStatus.APPROVED, decided_by="fake-approver")

    def send_alert(self, message: str, correlation_id: str) -> None:
        """Capture *(message, correlation_id)* for later assertion."""
        self.alerts.append((message, correlation_id))
