"""FrozenMarketData — MarketDataSource backed by committed CSV bar files.

Loads all ``*.csv`` files from *data_dir* at construction time.  Bar lookup
is O(1) by date for daily candles; range queries are O(n) over the small
per-symbol slice.

CSV format (one file per ticker):
    date,open,high,low,close,volume
    2024-10-21,230.15,232.48,...

The ``date`` column is parsed as a UTC midnight timestamp so callers can
pass ``datetime(2024, 10, 21, tzinfo=timezone.utc)`` or the equivalent
naive datetime and get exact matches.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from firm.domain import Bar


def _parse_bar(symbol: str, row: dict[str, str]) -> Bar:
    """Build a :class:`Bar` from a CSV row dict."""
    parsed_date = date.fromisoformat(row["date"])
    ts = datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
        tzinfo=UTC,
    )
    return Bar(
        symbol=symbol,
        open=Decimal(row["open"]),
        high=Decimal(row["high"]),
        low=Decimal(row["low"]),
        close=Decimal(row["close"]),
        volume=int(row["volume"]),
        ts=ts,
    )


def _load_csv(path: Path) -> list[Bar]:
    """Parse a single CSV file and return its bars in ascending timestamp order."""
    symbol = path.stem.upper()
    bars: list[Bar] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            bars.append(_parse_bar(symbol, row))
    bars.sort(key=lambda b: b.ts)
    return bars


def _normalise_ts(ts: datetime) -> datetime:
    """Return a UTC midnight datetime suitable for date-level exact matching.

    If *ts* is already midnight UTC it is returned as-is.  Otherwise the
    calendar date is extracted and midnight UTC is returned so callers using
    intra-day timestamps still hit daily bars correctly.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return datetime(ts.year, ts.month, ts.day, tzinfo=UTC)


class FrozenMarketData:
    """Implements :class:`~firm.ports.market_data.MarketDataSource`.

    All bar data is loaded eagerly at construction time; subsequent queries
    are pure in-memory operations with no IO.
    """

    def __init__(self, data_dir: Path) -> None:
        self._bars: dict[str, list[Bar]] = {}
        self._index: dict[str, dict[datetime, Bar]] = {}
        for csv_path in sorted(data_dir.glob("*.csv")):
            symbol = csv_path.stem.upper()
            bars = _load_csv(csv_path)
            self._bars[symbol] = bars
            self._index[symbol] = {bar.ts: bar for bar in bars}

    # ------------------------------------------------------------------
    # MarketDataSource protocol
    # ------------------------------------------------------------------

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        """Return the daily bar for *symbol* whose date matches *ts*.

        Performs a date-level match so an intra-day timestamp like
        ``2024-10-23T14:30:00Z`` resolves to the Oct-23 bar, consistent with
        the frozen daily-resolution data.
        """
        sym_index = self._index.get(symbol.upper())
        if sym_index is None:
            return None
        normalised = _normalise_ts(ts)
        return sym_index.get(normalised)

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Return bars for *symbol* in the half-open interval ``[start, end)``.

        Both *start* and *end* are normalised to UTC midnight for date-level
        comparison before filtering.
        """
        bars = self._bars.get(symbol.upper(), [])
        norm_start = _normalise_ts(start)
        norm_end = _normalise_ts(end)
        return [b for b in bars if norm_start <= b.ts < norm_end]
