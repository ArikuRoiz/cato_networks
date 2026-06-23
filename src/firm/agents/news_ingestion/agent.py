"""NewsIngestionAgent — fetch live headlines via yfinance and store in pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from firm.agents.base import BaseAgent
from firm.agents.news_ingestion.schemas import (
    NewsIngested,
    NewsIngestionFailure,
    NewsIngestionInput,
)
from firm.ports.evidence import EvidenceStore
from firm.ports.types import NewsDoc

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawItem:
    """Normalised view of one yfinance news item — handles both schema shapes."""

    title: str
    url: str
    publisher: str
    pub_date_raw: str | int | None  # ISO string (new) or epoch int (legacy)
    summary: str

    @property
    def is_valid(self) -> bool:
        return bool(self.title and self.url and self.pub_date_raw is not None)

    @property
    def published_at(self) -> datetime | None:
        if self.pub_date_raw is None:
            return None
        if isinstance(self.pub_date_raw, str):
            try:
                return datetime.fromisoformat(self.pub_date_raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        try:
            return datetime.fromtimestamp(int(self.pub_date_raw), tz=UTC)
        except (ValueError, OSError):
            return None

    @property
    def body(self) -> str:
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        if self.publisher:
            parts.append(f"Source: {self.publisher}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validators / helpers
# ---------------------------------------------------------------------------


def _ingest_symbol(
    evidence: EvidenceStore, symbol: str, cutoff: datetime
) -> int | NewsIngestionFailure:
    try:
        yf = _import_yfinance()
    except ImportError:
        return NewsIngestionFailure(
            reason="yfinance not installed; run: pip install yfinance", symbol=symbol
        )

    try:
        ticker = yf.Ticker(symbol)
        raw_news: list[Any] = ticker.news or []
    except Exception as exc:
        return NewsIngestionFailure(reason=f"yfinance fetch failed: {exc}", symbol=symbol)

    count = 0
    for raw in raw_news:
        doc = _to_news_doc(symbol, raw, cutoff)
        if doc is None:
            continue
        try:
            evidence.embed_and_store(doc)
            count += 1
        except Exception:
            continue

    return count


def _parse_raw_item(raw: dict[str, Any]) -> _RawItem:
    """Wrap one yfinance news dict — handles nested `content` (new) and flat (legacy)."""
    content: dict[str, Any] = raw.get("content") or {}
    if content:
        # New nested schema (yfinance ≥ 0.2.x, circa 2025+)
        canonical: dict[str, Any] = content.get("canonicalUrl") or {}
        provider: dict[str, Any] = content.get("provider") or {}
        return _RawItem(
            title=content.get("title", "").strip(),
            url=(canonical.get("url") or "").strip(),
            publisher=(provider.get("displayName") or "").strip(),
            pub_date_raw=content.get("pubDate") or content.get("displayTime"),
            summary=content.get("summary", "").strip(),
        )
    # Legacy flat schema
    return _RawItem(
        title=raw.get("title", "").strip(),
        url=raw.get("link", "").strip(),
        publisher=raw.get("publisher", "").strip(),
        pub_date_raw=raw.get("providerPublishTime"),
        summary="",
    )


def _to_news_doc(symbol: str, raw: dict[str, Any], cutoff: datetime) -> NewsDoc | None:
    item = _parse_raw_item(raw)
    if not item.is_valid:
        return None

    published_at = item.published_at
    if published_at is None or published_at < cutoff:
        return None

    return NewsDoc(
        symbol=symbol,
        text=item.body,
        source_url=item.url,
        published_at=published_at,
    )


def _import_yfinance() -> Any:
    """Import and return the yfinance module; raise ImportError with hint if absent."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]  # no stubs

        return yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for news ingestion. Run: pip install yfinance"
        ) from exc
