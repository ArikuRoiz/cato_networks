"""Unit tests for NewsIngestionAgent — no live network, yfinance mocked.

Covers:
- New nested-content schema (yfinance >= 0.2.x, 2025+)
- Legacy flat schema (backward-compat)
- Cutoff filtering (articles older than lookback_hours are dropped)
- Missing required fields → skipped, no crash
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from firm.adapters.fakes import FakeEvidenceStore
from firm.agents.news_ingestion.agent import (
    NewsIngestionAgent,
    _parse_raw_item,
    _to_news_doc,
)
from firm.agents.news_ingestion.schemas import (
    NewsIngested,
    NewsIngestionFailure,
    NewsIngestionInput,
)
from firm.ports.types import NewsDoc

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 23, 12, 0, 0, tzinfo=UTC)
_RECENT = (_NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")  # 1 h ago (within 24h)
_OLD = (_NOW - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")  # 48 h ago (outside default 24h)


def _nested_item(
    title: str = "NVDA beats estimates",
    url: str = "https://finance.yahoo.com/news/nvda",
    publisher: str = "Yahoo Finance",
    pub_date: str = _RECENT,
    summary: str = "Strong data-centre revenue drove the beat.",
) -> dict:
    """Build a yfinance news item in the new nested-content schema."""
    return {
        "id": "abc-123",
        "content": {
            "id": "abc-123",
            "contentType": "STORY",
            "title": title,
            "summary": summary,
            "pubDate": pub_date,
            "displayTime": pub_date,
            "provider": {"displayName": publisher},
            "canonicalUrl": {"url": url},
            "clickThroughUrl": {"url": url},
        },
    }


def _legacy_item(
    title: str = "NVDA beats estimates",
    link: str = "https://finance.yahoo.com/news/nvda-legacy",
    publisher: str = "Reuters",
    epoch: int | None = None,
) -> dict:
    """Build a yfinance news item in the old flat schema."""
    if epoch is None:
        epoch = int((_NOW - timedelta(hours=2)).timestamp())
    return {
        "title": title,
        "link": link,
        "publisher": publisher,
        "providerPublishTime": epoch,
    }


# ---------------------------------------------------------------------------
# _parse_raw_item tests
# ---------------------------------------------------------------------------


class TestParseRawItem:
    def test_parses_nested_schema(self) -> None:
        item = _parse_raw_item(_nested_item())
        assert item.title == "NVDA beats estimates"
        assert item.url == "https://finance.yahoo.com/news/nvda"
        assert item.publisher == "Yahoo Finance"
        assert item.summary == "Strong data-centre revenue drove the beat."
        assert item.pub_date_raw == _RECENT

    def test_parses_legacy_schema(self) -> None:
        epoch = int((_NOW - timedelta(hours=2)).timestamp())
        item = _parse_raw_item(_legacy_item(epoch=epoch))
        assert item.title == "NVDA beats estimates"
        assert item.url == "https://finance.yahoo.com/news/nvda-legacy"
        assert item.publisher == "Reuters"
        assert isinstance(item.pub_date_raw, int)

    def test_nested_published_at_parses_iso_string(self) -> None:
        item = _parse_raw_item(_nested_item(pub_date="2026-06-22T10:00:00Z"))
        ts = item.published_at
        assert ts is not None
        assert ts.tzinfo is not None
        assert ts.year == 2026

    def test_legacy_published_at_parses_epoch(self) -> None:
        epoch = int((_NOW - timedelta(hours=3)).timestamp())
        item = _parse_raw_item(_legacy_item(epoch=epoch))
        ts = item.published_at
        assert ts is not None
        assert ts.tzinfo is not None

    def test_is_valid_true_for_complete_item(self) -> None:
        assert _parse_raw_item(_nested_item()).is_valid

    def test_is_valid_false_when_title_missing(self) -> None:
        item = _parse_raw_item(_nested_item(title=""))
        assert not item.is_valid

    def test_is_valid_false_when_url_missing(self) -> None:
        item = _parse_raw_item(_nested_item(url=""))
        assert not item.is_valid

    def test_body_includes_title_summary_and_publisher(self) -> None:
        item = _parse_raw_item(_nested_item())
        assert "NVDA beats estimates" in item.body
        assert "Strong data-centre revenue" in item.body
        assert "Yahoo Finance" in item.body

    def test_body_without_summary_still_includes_title(self) -> None:
        item = _parse_raw_item(_nested_item(summary=""))
        assert "NVDA beats estimates" in item.body


# ---------------------------------------------------------------------------
# _to_news_doc tests
# ---------------------------------------------------------------------------


class TestToNewsDoc:
    def test_nested_schema_produces_news_doc(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _nested_item(), cutoff)
        assert isinstance(doc, NewsDoc)
        assert doc.symbol == "NVDA"
        assert doc.source_url == "https://finance.yahoo.com/news/nvda"
        assert doc.published_at.tzinfo is not None

    def test_legacy_schema_produces_news_doc(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _legacy_item(), cutoff)
        assert isinstance(doc, NewsDoc)
        assert doc.source_url == "https://finance.yahoo.com/news/nvda-legacy"

    def test_article_older_than_cutoff_is_dropped(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _nested_item(pub_date=_OLD), cutoff)
        assert doc is None

    def test_article_within_cutoff_is_kept(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _nested_item(pub_date=_RECENT), cutoff)
        assert doc is not None

    def test_item_missing_title_is_dropped(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _nested_item(title=""), cutoff)
        assert doc is None

    def test_item_missing_url_is_dropped(self) -> None:
        cutoff = _NOW - timedelta(hours=24)
        doc = _to_news_doc("NVDA", _nested_item(url=""), cutoff)
        assert doc is None


# ---------------------------------------------------------------------------
# NewsIngestionAgent integration (mocked yfinance)
# ---------------------------------------------------------------------------


class TestNewsIngestionAgent:
    def _make_agent(self) -> tuple[NewsIngestionAgent, FakeEvidenceStore]:
        store = FakeEvidenceStore()
        return NewsIngestionAgent(evidence=store), store

    def _make_input(
        self, symbols: list[str] | None = None, lookback_hours: int = 24
    ) -> NewsIngestionInput:
        return NewsIngestionInput(
            symbols=symbols or ["NVDA"],
            lookback_hours=lookback_hours,
            correlation_id="test-corr-id",
        )

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_nested_schema_articles_are_stored(self, mock_import: MagicMock) -> None:
        """New nested-content schema → NewsDocs produced and stored."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_ticker.news = [
            _nested_item(),
            _nested_item(title="NVDA H100 demand surge", url="https://example.com/h100"),
        ]
        mock_yf.Ticker.return_value = mock_ticker

        agent, store = self._make_agent()
        result = agent.run(self._make_input())

        assert isinstance(result, NewsIngested)
        assert result.articles_added == 2
        assert "NVDA" in result.symbols_updated
        assert len(store.docs) == 2

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_legacy_schema_articles_are_stored(self, mock_import: MagicMock) -> None:
        """Old flat schema still ingested (back-compat)."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        epoch = int((_NOW - timedelta(hours=2)).timestamp())
        mock_ticker = MagicMock()
        mock_ticker.news = [_legacy_item(epoch=epoch)]
        mock_yf.Ticker.return_value = mock_ticker

        agent, store = self._make_agent()
        result = agent.run(self._make_input())

        assert isinstance(result, NewsIngested)
        assert result.articles_added == 1
        assert len(store.docs) == 1

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_stale_articles_are_filtered(self, mock_import: MagicMock) -> None:
        """Articles older than lookback_hours are silently dropped."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_ticker.news = [_nested_item(pub_date=_OLD)]  # 48h ago, outside 24h window
        mock_yf.Ticker.return_value = mock_ticker

        agent, _store = self._make_agent()
        result = agent.run(self._make_input(lookback_hours=24))

        assert isinstance(result, NewsIngested)
        assert result.articles_added == 0
        assert result.symbols_updated == []

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_yfinance_exception_returns_failure(self, mock_import: MagicMock) -> None:
        """yfinance fetch error → NewsIngestionFailure (no exception raised)."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_yf.Ticker.side_effect = RuntimeError("Yahoo Finance unavailable")

        agent, _ = self._make_agent()
        result = agent.run(self._make_input())

        assert isinstance(result, NewsIngestionFailure)
        assert "Yahoo Finance unavailable" in result.reason

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_empty_news_returns_zero_added(self, mock_import: MagicMock) -> None:
        """Empty news list → NewsIngested with 0 articles added."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        agent, _store = self._make_agent()
        result = agent.run(self._make_input())

        assert isinstance(result, NewsIngested)
        assert result.articles_added == 0
        assert result.symbols_updated == []

    @patch("firm.agents.news_ingestion.agent._import_yfinance")
    def test_multiple_symbols_aggregated(self, mock_import: MagicMock) -> None:
        """Multiple symbols each get articles; totals are summed."""
        mock_yf = MagicMock()
        mock_import.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_ticker.news = [_nested_item()]
        mock_yf.Ticker.return_value = mock_ticker

        agent, _store = self._make_agent()
        result = agent.run(self._make_input(symbols=["NVDA", "AMD"]))

        assert isinstance(result, NewsIngested)
        assert result.articles_added == 2
        assert set(result.symbols_updated) == {"NVDA", "AMD"}
