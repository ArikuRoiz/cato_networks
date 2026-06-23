"""NYSECalendar — market-hours gating backed by pandas_market_calendars.

The single responsibility of this module is to answer two questions:
  1. Is the NYSE market open at a given UTC timestamp?
  2. When does the NYSE next open after a given UTC timestamp?

All datetime arithmetic is done in US/Eastern to match NYSE rules.  Input
timestamps are assumed to be timezone-aware UTC; the helpers enforce this.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

_EASTERN = ZoneInfo("US/Eastern")
_UTC = UTC

# Regular session: 09:30-16:00 ET (inclusive open, exclusive close boundary
# handled by the seconds-precision checks in is_market_open).
_OPEN_HOUR = 9
_OPEN_MINUTE = 30

# pandas_market_calendars schedule() requires a date-string window.
# We fetch a two-week lookahead so next_open() can always find an answer
# without iterating calendar calls.
_NEXT_OPEN_LOOKAHEAD_DAYS = 14


def _to_eastern(ts: datetime) -> datetime:
    """Convert a UTC-aware datetime to US/Eastern, raising if naive."""
    if ts.tzinfo is None:
        raise ValueError(
            f"Timezone-naive datetime passed to NYSECalendar: {ts!r}. "
            "All timestamps must be timezone-aware UTC."
        )
    return ts.astimezone(_EASTERN)


class NYSECalendar:
    """NYSE market-hours gating.

    Wraps ``pandas_market_calendars`` to provide simple, typed queries.
    A single instance can be shared across the application; the underlying
    calendar object is stateless and thread-safe.
    """

    def __init__(self) -> None:
        self._cal = mcal.get_calendar("NYSE")

    def _get_schedule(self, start: str, end: str) -> pd.DataFrame:
        """Fetch the NYSE schedule DataFrame for the given date-string range."""
        return self._cal.schedule(start_date=start, end_date=end)

    def is_market_open(self, ts: datetime) -> bool:
        """Return ``True`` iff the NYSE is open at *ts*.

        Handles:
        - Weekends → False
        - NYSE holidays → False
        - Early-close half-days (e.g. day after Thanksgiving, closes 13:00 ET) → False
          after the early-close time
        - Before 09:30 ET → False
        - At or after 16:00 ET → False
        - DST boundary correctness (handled by ZoneInfo conversion)

        *ts* must be timezone-aware UTC.
        """
        et = _to_eastern(ts)
        date_str = et.strftime("%Y-%m-%d")
        schedule = self._get_schedule(date_str, date_str)
        if schedule.empty:
            return False
        return _is_within_session(et, schedule.iloc[0])

    def next_open(self, ts: datetime) -> datetime:
        """Return the next NYSE market open after *ts* as a UTC datetime.

        Scans forward up to ``_NEXT_OPEN_LOOKAHEAD_DAYS`` trading days.
        Raises ``RuntimeError`` if no open is found within the lookahead
        window (should not happen in normal usage).

        *ts* must be timezone-aware UTC.
        """
        et = _to_eastern(ts)
        start_str = et.strftime("%Y-%m-%d")
        end_date = et + timedelta(days=_NEXT_OPEN_LOOKAHEAD_DAYS)
        end_str = end_date.strftime("%Y-%m-%d")

        schedule = self._get_schedule(start_str, end_str)
        return _find_next_open(et, schedule)


def _is_within_session(et: datetime, row: pd.Series) -> bool:
    """Check whether *et* falls within the session described by *row*.

    *row* comes from the mcal schedule DataFrame and has ``market_open`` and
    ``market_close`` columns as UTC-aware Timestamps.  We convert them to ET
    for the comparison so DST is handled correctly.
    """
    session_open: datetime = row["market_open"].to_pydatetime().astimezone(_EASTERN)
    session_close: datetime = row["market_close"].to_pydatetime().astimezone(_EASTERN)

    # Assert mcal returns the expected 09:30 ET open so any deviation is visible
    # rather than silently clamped.
    assert session_open.hour == _OPEN_HOUR, (
        f"Unexpected mcal session open hour: {session_open!r} — expected 09:30 ET"
    )
    assert session_open.minute == _OPEN_MINUTE, (
        f"Unexpected mcal session open minute: {session_open!r} — expected 09:30 ET"
    )
    # Early-close half-days: session_close will be < 16:00 ET.
    # Full sessions: session_close will be 16:00 ET.
    return session_open <= et < session_close


def _find_next_open(et: datetime, schedule: pd.DataFrame) -> datetime:
    """Return the first session open that is strictly after *et*."""
    for _, row in schedule.iterrows():
        session_open: datetime = row["market_open"].to_pydatetime().astimezone(_EASTERN)
        if session_open > et:
            return session_open.astimezone(_UTC)
    raise RuntimeError(f"No NYSE open found within {_NEXT_OPEN_LOOKAHEAD_DAYS} days of {et!r}")
