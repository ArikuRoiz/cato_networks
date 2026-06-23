"""Unit tests for the in-memory fake port implementations.

All tests are pure-Python — no IO, no DB, no network.
Covers:
- FakeEvidenceStore no-lookahead filtering
- FakeLLM sequential response replay and overflow guard
- FakeMarketData range queries
- FakeReportSink send_daily_report / send_hitl_request / send_alert
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from firm.adapters.fakes import (
    FakeEvidenceStore,
    FakeLLM,
    FakeMarketData,
    FakeReportSink,
)
from firm.domain import Bar
from firm.ports.evidence import EvidenceStore
from firm.ports.llm import LLM
from firm.ports.market_data import MarketDataSource
from firm.ports.report import ReportSink
from firm.ports.types import (
    Chunk,
    DailyReport,
    HITLRequest,
    LLMMessage,
    LLMResponse,
    NewsDoc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 10, 21, 9, 30, 0, tzinfo=UTC)


def _ts(offset_minutes: int = 0) -> datetime:
    return _BASE_TS + timedelta(minutes=offset_minutes)


def _bar(symbol: str, close: str, ts: datetime | None = None) -> Bar:
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
        ts=ts or _BASE_TS,
    )


def _chunk(symbol: str, published_at: datetime, text: str = "news") -> Chunk:
    return Chunk(
        id=uuid4(),
        symbol=symbol,
        text=text,
        source_url="https://example.com",
        chunk_id=f"chunk-{uuid4().hex[:8]}",
        published_at=published_at,
        score=0.8,
    )


def _llm_response(content: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        model="claude-haiku-test",
    )


# ---------------------------------------------------------------------------
# Protocol compliance checks
# ---------------------------------------------------------------------------


def test_fake_market_data_satisfies_protocol() -> None:
    """FakeMarketData is recognised as a MarketDataSource at runtime."""
    fake = FakeMarketData()
    assert isinstance(fake, MarketDataSource)


def test_fake_evidence_store_satisfies_protocol() -> None:
    """FakeEvidenceStore is recognised as an EvidenceStore at runtime."""
    fake = FakeEvidenceStore()
    assert isinstance(fake, EvidenceStore)


def test_fake_llm_satisfies_protocol() -> None:
    """FakeLLM is recognised as an LLM at runtime."""
    fake = FakeLLM(responses=[_llm_response()])
    assert isinstance(fake, LLM)


def test_fake_report_sink_satisfies_protocol() -> None:
    """FakeReportSink is recognised as a ReportSink at runtime."""
    fake = FakeReportSink()
    assert isinstance(fake, ReportSink)


# ---------------------------------------------------------------------------
# FakeMarketData
# ---------------------------------------------------------------------------


def test_fake_market_data_get_bar_hit() -> None:
    """get_bar returns the bar when it exists."""
    fake = FakeMarketData()
    bar = _bar("AAPL", "150")
    fake.add_bar(bar)

    result = fake.get_bar("AAPL", _BASE_TS)

    assert result == bar


def test_fake_market_data_get_bar_miss() -> None:
    """get_bar returns None for an unknown (symbol, ts) pair."""
    fake = FakeMarketData()
    result = fake.get_bar("AAPL", _BASE_TS)
    assert result is None


def test_fake_market_data_get_bars_returns_bars_in_range() -> None:
    """get_bars returns only bars whose ts falls in [start, end)."""
    fake = FakeMarketData()
    bar_early = _bar("NVDA", "100", ts=_ts(0))
    bar_mid = _bar("NVDA", "101", ts=_ts(5))
    bar_late = _bar("NVDA", "102", ts=_ts(10))
    for b in (bar_early, bar_mid, bar_late):
        fake.add_bar(b)

    results = fake.get_bars("NVDA", start=_ts(0), end=_ts(10))

    assert len(results) == 2
    assert results[0] == bar_early
    assert results[1] == bar_mid


def test_fake_market_data_get_bars_excludes_end_boundary() -> None:
    """get_bars uses a half-open interval — bar at ts==end is excluded."""
    fake = FakeMarketData()
    bar = _bar("MSFT", "300", ts=_ts(10))
    fake.add_bar(bar)

    results = fake.get_bars("MSFT", start=_ts(0), end=_ts(10))

    assert results == []


def test_fake_market_data_get_bars_wrong_symbol() -> None:
    """get_bars ignores bars for other symbols."""
    fake = FakeMarketData()
    fake.add_bar(_bar("AAPL", "150", ts=_ts(0)))

    results = fake.get_bars("NVDA", start=_ts(0), end=_ts(60))

    assert results == []


def test_fake_market_data_get_bars_sorted_by_ts() -> None:
    """get_bars returns bars in ascending ts order regardless of insertion order."""
    fake = FakeMarketData()
    bar_later = _bar("AMD", "90", ts=_ts(5))
    bar_earlier = _bar("AMD", "80", ts=_ts(0))
    fake.add_bar(bar_later)
    fake.add_bar(bar_earlier)

    results = fake.get_bars("AMD", start=_ts(0), end=_ts(60))

    assert results[0].ts < results[1].ts


# ---------------------------------------------------------------------------
# FakeEvidenceStore — no-lookahead filtering
# ---------------------------------------------------------------------------


def test_fake_evidence_store_filters_by_published_at() -> None:
    """search returns only chunks published at or before 'before'."""
    fake = FakeEvidenceStore()
    old = _chunk("NVDA", published_at=_ts(-60))
    exact = _chunk("NVDA", published_at=_ts(0))
    future = _chunk("NVDA", published_at=_ts(60))
    fake.docs.extend([old, exact, future])

    results = fake.search("NVDA", before=_ts(0))

    assert len(results) == 2
    result_ids = {c.id for c in results}
    assert old.id in result_ids
    assert exact.id in result_ids
    assert future.id not in result_ids


def test_fake_evidence_store_no_lookahead() -> None:
    """search never returns a chunk with published_at > before (no lookahead)."""
    fake = FakeEvidenceStore()
    future_chunk = _chunk("AAPL", published_at=_ts(1))
    fake.docs.append(future_chunk)

    results = fake.search("AAPL", before=_ts(0))

    assert results == []


def test_fake_evidence_store_filters_by_symbol() -> None:
    """search returns only chunks for the requested symbol."""
    fake = FakeEvidenceStore()
    nvda = _chunk("NVDA", published_at=_ts(-10))
    aapl = _chunk("AAPL", published_at=_ts(-10))
    fake.docs.extend([nvda, aapl])

    results = fake.search("NVDA", before=_ts(0))

    assert all(c.symbol == "NVDA" for c in results)
    assert len(results) == 1


def test_fake_evidence_store_respects_k_limit() -> None:
    """search returns at most k results."""
    fake = FakeEvidenceStore()
    for i in range(20):
        fake.docs.append(_chunk("META", published_at=_ts(-i - 1)))

    results = fake.search("META", before=_ts(0), k=5)

    assert len(results) == 5


def test_fake_evidence_store_embed_and_store() -> None:
    """embed_and_store creates a Chunk from a NewsDoc and appends it to docs."""
    fake = FakeEvidenceStore()
    doc = NewsDoc(
        symbol="GOOGL",
        text="Google Q3 earnings beat.",
        source_url="https://example.com/googl",
        published_at=_ts(-30),
    )

    fake.embed_and_store(doc)

    assert len(fake.docs) == 1
    stored = fake.docs[0]
    assert stored.symbol == "GOOGL"
    assert stored.text == doc.text
    assert stored.source_url == doc.source_url
    assert stored.published_at == doc.published_at
    assert stored.score == 0.0


def test_fake_evidence_store_embed_and_store_then_searchable() -> None:
    """A doc stored via embed_and_store is immediately searchable."""
    fake = FakeEvidenceStore()
    doc = NewsDoc(
        symbol="SPY",
        text="Market summary.",
        source_url="https://example.com/spy",
        published_at=_ts(-5),
    )
    fake.embed_and_store(doc)

    results = fake.search("SPY", before=_ts(0))

    assert len(results) == 1
    assert results[0].text == doc.text


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


def test_fake_llm_returns_responses_in_order() -> None:
    """complete returns pre-loaded responses in the order they were provided."""
    r1 = _llm_response("first")
    r2 = _llm_response("second")
    fake = FakeLLM(responses=[r1, r2])
    msgs = [LLMMessage(role="user", content="hello")]

    first = fake.complete(msgs, model="haiku", max_tokens=100)
    second = fake.complete(msgs, model="haiku", max_tokens=100)

    assert isinstance(first, LLMResponse)
    assert isinstance(second, LLMResponse)
    assert first.content == "first"
    assert second.content == "second"


def test_fake_llm_raises_on_overflow() -> None:
    """complete raises IndexError when all responses have been consumed."""
    fake = FakeLLM(responses=[_llm_response()])
    msgs = [LLMMessage(role="user", content="q")]
    fake.complete(msgs, model="haiku", max_tokens=100)  # consumes the only response

    with pytest.raises(IndexError):
        fake.complete(msgs, model="haiku", max_tokens=100)


def test_fake_llm_count_tokens_rough_estimate() -> None:
    """count_tokens returns total content length divided by 4."""
    fake = FakeLLM()
    msgs = [
        LLMMessage(role="user", content="abcd"),  # 4 chars
        LLMMessage(role="system", content="efgh"),  # 4 chars
    ]

    count = fake.count_tokens(msgs, model="haiku")

    assert count == 2  # 8 chars // 4


def test_fake_llm_index_advances_correctly() -> None:
    """Index increments for each complete call consumed."""
    responses = [_llm_response(f"r{i}") for i in range(3)]
    fake = FakeLLM(responses=responses)
    msgs = [LLMMessage(role="user", content="x")]

    for i in range(3):
        result = fake.complete(msgs, model="haiku", max_tokens=100)
        assert isinstance(result, LLMResponse)
        assert result.content == f"r{i}"

    assert fake.index == 3


# ---------------------------------------------------------------------------
# FakeReportSink
# ---------------------------------------------------------------------------


def test_fake_report_sink_send_daily_report() -> None:
    """send_daily_report appends the report to daily_reports_sent."""
    import datetime as dt

    fake = FakeReportSink()
    report = DailyReport(
        date=dt.date(2024, 10, 21),
        nav=Decimal("100000"),
        pnl=Decimal("500"),
        benchmark_return=0.01,
        trades=[],
        positions=[],
        citations=[],
    )
    fake.send_daily_report(report)

    assert len(fake.daily_reports_sent) == 1
    assert fake.daily_reports_sent[0] == report


def test_fake_report_sink_send_hitl_request_auto_approves() -> None:
    """send_hitl_request captures the request and returns an approved result."""
    fake = FakeReportSink()
    req = HITLRequest(
        trade_id=uuid4(),
        symbol="NVDA",
        side="buy",
        qty_str="100",
        notional=Decimal("13500"),
        reason="Exceeds HITL threshold",
        expires_at=_ts(15),
        correlation_id="corr-abc123",
    )

    result = fake.send_hitl_request(req)

    assert result.status == "approved"
    assert len(fake.hitl_requests) == 1
    assert fake.hitl_requests[0] == req


def test_fake_report_sink_send_alert_captures() -> None:
    """send_alert captures message and correlation_id."""
    fake = FakeReportSink()

    fake.send_alert("circuit breaker tripped", "corr-xyz")

    assert len(fake.alerts) == 1
    msg, cid = fake.alerts[0]
    assert msg == "circuit breaker tripped"
    assert cid == "corr-xyz"
