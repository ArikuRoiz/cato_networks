"""Unit tests for LiveMarketData — all yfinance calls are monkeypatched.

No real network access occurs in any test.  The yfinance module is replaced
with a lightweight fake that returns either a minimal DataFrame-like object
or raises an exception so we can exercise every branch without network I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from firm.domain import Bar

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_df_row(
    *,
    date: str,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 102.0,
    volume: int = 1_000_000,
) -> tuple[object, dict[str, object]]:
    """Return a (timestamp, row_dict) pair that mimics a pandas iterrows tuple."""
    import pandas as pd

    ts = pd.Timestamp(date, tz="UTC")
    row = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }
    return ts, row


def _make_fake_df(*rows: tuple[str, dict[str, object]]) -> MagicMock:
    """Build a MagicMock that behaves like a pandas DataFrame with iterrows().

    Each element of *rows* is a ``(date_str, row_dict)`` pair.
    """
    import pandas as pd

    mock_df = MagicMock()
    iter_data = []
    for date_str, row_dict in rows:
        ts = pd.Timestamp(date_str, tz="UTC")
        row_series = MagicMock()
        row_series.to_dict.return_value = row_dict
        iter_data.append((ts, row_series))
    mock_df.iterrows.return_value = iter(iter_data)
    return mock_df


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _make_ticker_mock(df: MagicMock) -> MagicMock:
    ticker = MagicMock()
    ticker.history.return_value = df
    return ticker


# ---------------------------------------------------------------------------
# Tests for get_bars
# ---------------------------------------------------------------------------


class TestGetBars:
    def test_returns_bars_within_range(self) -> None:
        """get_bars returns only bars in [start, end)."""
        df = _make_fake_df(
            ("2024-10-21", {"Open": 130.0, "High": 135.0, "Low": 128.0, "Close": 132.0, "Volume": 500_000}),
            ("2024-10-22", {"Open": 132.0, "High": 138.0, "Low": 131.0, "Close": 136.0, "Volume": 600_000}),
            ("2024-10-23", {"Open": 136.0, "High": 145.0, "Low": 135.0, "Close": 143.0, "Volume": 900_000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bars = adapter.get_bars("NVDA", _utc(2024, 10, 21), _utc(2024, 10, 23))

        # end is exclusive: 10-23 is excluded
        assert len(bars) == 2
        assert all(isinstance(b, Bar) for b in bars)
        assert bars[0].ts == _utc(2024, 10, 21)
        assert bars[1].ts == _utc(2024, 10, 22)

    def test_returns_empty_when_no_data(self) -> None:
        df = _make_fake_df()
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bars = adapter.get_bars("AAPL", _utc(2024, 1, 1), _utc(2024, 1, 5))

        assert bars == []

    def test_bar_fields_mapped_correctly(self) -> None:
        df = _make_fake_df(
            ("2024-10-21", {"Open": 130.5, "High": 135.25, "Low": 128.75, "Close": 133.0, "Volume": 456_789}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bars = adapter.get_bars("NVDA", _utc(2024, 10, 21), _utc(2024, 10, 22))

        assert len(bars) == 1
        bar = bars[0]
        assert bar.symbol == "NVDA"
        assert bar.open == Decimal("130.5")
        assert bar.high == Decimal("135.25")
        assert bar.low == Decimal("128.75")
        assert bar.close == Decimal("133.0")
        assert bar.volume == 456_789

    def test_symbol_uppercased(self) -> None:
        df = _make_fake_df(
            ("2024-10-21", {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Volume": 1000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bars = adapter.get_bars("aapl", _utc(2024, 10, 21), _utc(2024, 10, 22))

        assert bars[0].symbol == "AAPL"

    def test_skips_rows_with_nan_close(self) -> None:
        import math

        df = _make_fake_df(
            ("2024-10-21", {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": math.nan, "Volume": 1000}),
            ("2024-10-22", {"Open": 101.0, "High": 106.0, "Low": 100.0, "Close": 103.0, "Volume": 2000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bars = adapter.get_bars("MSFT", _utc(2024, 10, 21), _utc(2024, 10, 23))

        assert len(bars) == 1
        assert bars[0].ts == _utc(2024, 10, 22)

    def test_raises_import_error_when_yfinance_missing(self) -> None:
        with patch(
            "firm.adapters.market_data_live._import_yfinance",
            side_effect=ImportError("yfinance is required for live market data."),
        ):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            with pytest.raises(ImportError, match="yfinance"):
                adapter.get_bars("NVDA", _utc(2024, 10, 21), _utc(2024, 10, 22))


# ---------------------------------------------------------------------------
# Tests for get_bar
# ---------------------------------------------------------------------------


class TestGetBar:
    def test_returns_bar_for_exact_date(self) -> None:
        df = _make_fake_df(
            ("2024-10-23", {"Open": 136.0, "High": 145.0, "Low": 135.0, "Close": 143.0, "Volume": 900_000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bar = adapter.get_bar("NVDA", _utc(2024, 10, 23))

        assert bar is not None
        assert bar.ts == _utc(2024, 10, 23)
        assert bar.close == Decimal("143.0")

    def test_returns_none_when_no_bar_for_date(self) -> None:
        df = _make_fake_df()
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bar = adapter.get_bar("NVDA", _utc(2024, 10, 26))  # weekend

        assert bar is None

    def test_intraday_ts_resolves_to_daily_bar(self) -> None:
        """An intra-day timestamp like 14:30 UTC still maps to the correct daily bar."""
        df = _make_fake_df(
            ("2024-10-23", {"Open": 136.0, "High": 145.0, "Low": 135.0, "Close": 143.0, "Volume": 900_000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            intraday = datetime(2024, 10, 23, 14, 30, 0, tzinfo=UTC)
            bar = adapter.get_bar("NVDA", intraday)

        assert bar is not None
        assert bar.ts == _utc(2024, 10, 23)

    def test_falls_back_to_last_bar_when_no_exact_match(self) -> None:
        """When the exact date is absent but another bar exists in the window, the last bar
        in the two-day fetch window is returned."""
        # yfinance returns the 10-24 bar when we query for 10-23 window (10-23 to 10-25).
        # This simulates a situation where the date requested has no bar but the next
        # day's bar is present.
        df = _make_fake_df(
            ("2024-10-24", {"Open": 140.0, "High": 145.0, "Low": 139.0, "Close": 143.0, "Volume": 700_000}),
        )
        ticker = _make_ticker_mock(df)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = ticker

        with patch("firm.adapters.market_data_live._import_yfinance", return_value=mock_yf):
            from firm.adapters.market_data_live import LiveMarketData

            adapter = LiveMarketData()
            bar = adapter.get_bar("NVDA", _utc(2024, 10, 23))

        # Falls back to the only available bar in the window (10-24)
        assert bar is not None
        assert bar.ts == _utc(2024, 10, 24)


# ---------------------------------------------------------------------------
# Tests for _import_yfinance helper
# ---------------------------------------------------------------------------


class TestImportYfinance:
    def test_raises_import_error_with_install_hint(self) -> None:
        with patch.dict("sys.modules", {"yfinance": None}):
            import importlib

            import firm.adapters.market_data_live as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="pip install yfinance"):
                mod._import_yfinance()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_live_market_data_satisfies_protocol(self) -> None:
        """LiveMarketData must be accepted by isinstance against MarketDataSource."""
        from firm.adapters.market_data_live import LiveMarketData
        from firm.ports.market_data import MarketDataSource

        assert isinstance(LiveMarketData(), MarketDataSource)
