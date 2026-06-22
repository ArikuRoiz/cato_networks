"""MarketDataSource port — the IO seam for market bar data.

Agents import this Protocol; adapters (live feed, frozen CSV, fake) implement it.
Domain core never depends on an adapter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from firm.domain import Bar


@runtime_checkable
class MarketDataSource(Protocol):
    """Read-only access to OHLCV bar data.

    Implementations must honour the ``runtime_checkable`` contract so fakes
    can be verified with ``isinstance`` in tests.
    """

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        """Return the bar for *symbol* at exactly *ts*, or ``None`` if absent."""
        ...

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Return all bars for *symbol* in the half-open interval [start, end)."""
        ...
