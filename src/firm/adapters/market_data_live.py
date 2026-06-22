"""LiveMarketData — placeholder adapter for a real-time market data feed.

This stub satisfies the :class:`~firm.ports.market_data.MarketDataSource`
Protocol so the type-checker accepts it anywhere a ``MarketDataSource`` is
expected.  All methods raise ``NotImplementedError`` at runtime to make it
obvious when un-wired code tries to use a live feed during tests or replay.

To connect a real feed:
1.  Replace this class body with the vendor-specific implementation.
2.  Inject ``LiveMarketData`` via the adapter layer — the domain core and
    agents depend only on the ``MarketDataSource`` port and are unaffected.
"""

from __future__ import annotations

from datetime import datetime

from firm.domain.entities import Bar

_NOT_IMPLEMENTED_MSG = "Live market feed not implemented — wire to real data source"


class LiveMarketData:
    """Implements :class:`~firm.ports.market_data.MarketDataSource`.

    Every method raises ``NotImplementedError`` unconditionally; this class
    exists purely so the type-checker verifies structural compatibility.
    """

    def get_bar(self, symbol: str, ts: datetime) -> Bar | None:
        """Raises ``NotImplementedError``; not yet wired to a live feed."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        """Raises ``NotImplementedError``; not yet wired to a live feed."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
