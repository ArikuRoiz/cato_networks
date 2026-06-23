"""LiveMarketData — MarketDataSource backed by yfinance.

Fetches real OHLCV bar data from Yahoo Finance via the ``yfinance`` library.
yfinance is a lazy import so the module loads without the package installed;
only the first ``get_bar`` / ``get_bars`` call will raise ``ImportError`` when
the library is absent.

Mapping:
    yfinance ``history()`` returns a pandas DataFrame with a DatetimeIndex
    (UTC-localised when ``tz_localize`` is applied) and columns Open, High,
    Low, Close, Volume.  Each row is mapped to a domain :class:`~firm.domain.Bar`.

Resolution:
    Daily bars (``interval="1d"``) are used to match the frozen-data contract.
    Intra-day timestamps are normalised to UTC midnight for ``get_bar`` lookups
    so callers using ``datetime.now(UTC)`` still resolve to the correct daily bar.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from firm.domain import Bar

# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class LiveMarketData:
    """Implements :class:`~firm.ports.market_data.MarketDataSource` via yfinance.

    yfinance is imported lazily on first use so that the rest of the codebase
    can import this module without the package installed (e.g. in CI where the
    offline suite runs without network dependencies).
    """

    # ------------------------------------------------------------------
    # MarketDataSource protocol
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Return daily bars for *symbol* in the half-open interval [start, end).

        The *end* date is included in the yfinance request (yfinance's ``end``
        is exclusive at day granularity) by requesting one extra day, then
        filtering at the domain level to honour the half-open contract.
        """
        yf: Any = _import_yfinance()
        ticker = yf.Ticker(symbol)
        # Add one day so yfinance's exclusive end includes our end date.
        yf_end = end + timedelta(days=1)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=yf_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        bars = _dataframe_to_bars(symbol, df)
        norm_start = _to_utc_midnight(start)
        norm_end = _to_utc_midnight(end)
        return [b for b in bars if norm_start <= b.ts < norm_end]

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        """Return the daily bar for *symbol* on the calendar date of *ts*.

        Fetches a two-day window centred on *ts* so the response is minimal.
        Returns ``None`` when yfinance returns no data for the date (e.g. a
        weekend or holiday).
        """
        date_ts = _to_utc_midnight(ts)
        start = date_ts
        # Request two days to cover the target date.
        end = date_ts + timedelta(days=2)
        bars = self.get_bars(symbol, start, end)
        # Return the bar whose date matches exactly; fall back to the last bar
        # in the window if no exact match (e.g. holiday shifted bar).
        for bar in bars:
            if bar.ts == date_ts:
                return bar
        return bars[-1] if bars else None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _import_yfinance() -> Any:
    """Import and return the yfinance module; raise ImportError with hint if absent."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]  # no stubs

        return yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for live market data. "
            "Install it with: pip install yfinance"
        ) from exc


def _to_utc_midnight(ts: datetime) -> datetime:
    """Normalise *ts* to UTC midnight (date-level resolution)."""
    utc = ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
    return datetime(utc.year, utc.month, utc.day, tzinfo=UTC)


def _dataframe_to_bars(symbol: str, df: Any) -> list[Bar]:
    """Convert a yfinance history DataFrame to a list of domain :class:`Bar` objects.

    Handles both tz-aware and tz-naive DatetimeIndex rows.  Rows where Close is
    NaN (market closure or split-adjustment artefacts) are silently skipped.
    """
    import math

    bars: list[Bar] = []
    for idx, row in _iter_df_rows(df):
        close = _safe_decimal(row.get("Close"))
        if close is None or (isinstance(close, float) and math.isnan(close)):
            continue
        bar = _row_to_bar(symbol, idx, row)
        if bar is not None:
            bars.append(bar)
    bars.sort(key=lambda b: b.ts)
    return bars


def _iter_df_rows(df: Any) -> Generator[tuple[Any, dict[str, Any]], None, None]:
    """Yield (index, row_dict) pairs from a pandas DataFrame.

    Accepts the DataFrame as ``Any`` (yfinance returns a pandas object with no
    stubs) so mypy is not involved in verifying the pandas internal API.
    """
    try:
        for idx, row in df.iterrows():
            yield idx, row.to_dict()
    except AttributeError:
        return


def _row_to_bar(symbol: str, idx: Any, row: dict[str, Any]) -> Bar | None:
    """Build a :class:`Bar` from a single DataFrame row; return ``None`` on bad data."""
    try:
        ts = _index_to_utc_midnight(idx)
        volume_raw = row.get("Volume")
        volume = int(volume_raw) if volume_raw is not None else 0
        return Bar(
            symbol=symbol.upper(),
            open=_safe_decimal(row.get("Open")) or Decimal("0"),
            high=_safe_decimal(row.get("High")) or Decimal("0"),
            low=_safe_decimal(row.get("Low")) or Decimal("0"),
            close=_safe_decimal(row.get("Close")) or Decimal("0"),
            volume=volume,
            ts=ts,
        )
    except Exception:
        return None


def _index_to_utc_midnight(idx: Any) -> datetime:
    """Convert a pandas Timestamp index to a UTC midnight datetime."""
    import pandas as pd

    if isinstance(idx, pd.Timestamp):
        ts: datetime = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return _to_utc_midnight(ts)
    raise TypeError(f"Unexpected index type: {type(idx)}")


def _safe_decimal(value: Any) -> Decimal | None:
    """Convert *value* to :class:`Decimal`, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
