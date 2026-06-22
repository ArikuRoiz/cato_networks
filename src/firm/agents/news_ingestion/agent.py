"""NewsIngestionAgent — fetch live headlines via yfinance and store in pgvector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from firm.agents.base import BaseAgent
from firm.agents.news_ingestion.schemas import (
    NewsIngested,
    NewsIngestionFailure,
    NewsIngestionInput,
)
from firm.ports.evidence import EvidenceStore
from firm.ports.types import NewsDoc


class NewsIngestionAgent(BaseAgent[NewsIngestionInput, NewsIngested | NewsIngestionFailure]):
    """Fetches recent headlines from Yahoo Finance (yfinance) and upserts them
    into the pgvector evidence store so the ResearchAgent always has fresh data.

    No API key required — yfinance scrapes Yahoo Finance's public feed.
    Articles already in the store are silently skipped (ON CONFLICT DO NOTHING).
    """

    def __init__(self, evidence: EvidenceStore) -> None:
        self._evidence = evidence

    def run(self, inp: NewsIngestionInput) -> NewsIngested | NewsIngestionFailure:
        cutoff = datetime.now(UTC) - timedelta(hours=inp.lookback_hours)
        added = 0
        updated: list[str] = []

        for symbol in inp.symbols:
            result = _ingest_symbol(self._evidence, symbol, cutoff)
            if isinstance(result, NewsIngestionFailure):
                return result
            added += result
            if result > 0:
                updated.append(symbol)

        return NewsIngested(articles_added=added, symbols_updated=updated)


def _ingest_symbol(
    evidence: EvidenceStore, symbol: str, cutoff: datetime
) -> int | NewsIngestionFailure:
    try:
        import yfinance as yf  # local import — optional dependency
    except ImportError:
        return NewsIngestionFailure(
            reason="yfinance not installed; run: pip install yfinance", symbol=symbol
        )

    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news or []
    except Exception as exc:
        return NewsIngestionFailure(reason=f"yfinance fetch failed: {exc}", symbol=symbol)

    count = 0
    for item in raw_news:
        doc = _to_news_doc(symbol, item, cutoff)
        if doc is None:
            continue
        try:
            evidence.embed_and_store(doc)
            count += 1
        except Exception:
            continue

    return count


def _to_news_doc(symbol: str, item: dict, cutoff: datetime) -> NewsDoc | None:  # type: ignore[type-arg]
    title: str = item.get("title", "").strip()
    link: str = item.get("link", "").strip()
    ts_raw = item.get("providerPublishTime")

    if not title or not link or ts_raw is None:
        return None

    published_at = datetime.fromtimestamp(int(ts_raw), tz=UTC)
    if published_at < cutoff:
        return None

    # Best-effort body: title + publisher for richer chunk text
    publisher: str = item.get("publisher", "")
    body = f"{title}. Source: {publisher}." if publisher else title

    return NewsDoc(
        symbol=symbol,
        text=body,
        source_url=link,
        published_at=published_at,
    )
