"""Project-wide named constants for values previously inlined in the CLI.

Centralising these keeps the watchlist, starting cash, and runtime defaults
in one place rather than duplicated across command handlers.
"""

from __future__ import annotations

from decimal import Decimal

DEFAULT_WATCHLIST: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD")
DEFAULT_INITIAL_CASH = Decimal("100000")
DEV_POLL_INTERVAL_SECONDS = 30
DEFAULT_WEB_PORT = 8000
DEFAULT_LOOKBACK_DAYS = 7
