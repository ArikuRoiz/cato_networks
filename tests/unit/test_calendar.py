"""Unit tests for NYSECalendar — market-hours gating.

Covers the test_market_calendar_gating requirement (FIRM-8):
  - NYSE holidays → is_market_open = False
  - Half-day early close (day after Thanksgiving 2024, closes 13:00 ET) → False
    when ts is 13:01 ET
  - Normal trading hours → True
  - Weekend → False
  - Boundary: 15:59:59 ET → True; 16:00:01 ET → False
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from firm.services.calendar import NYSECalendar

_ET = ZoneInfo("US/Eastern")
_UTC = UTC


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """Build a UTC-aware datetime for test clarity."""
    return datetime(year, month, day, hour, minute, second, tzinfo=_UTC)


def _et(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """Build an ET-aware datetime converted to UTC for test clarity."""
    et_dt = datetime(year, month, day, hour, minute, second, tzinfo=_ET)
    return et_dt.astimezone(_UTC)


@pytest.fixture(scope="module")
def calendar() -> NYSECalendar:
    return NYSECalendar()


def test_thanksgiving_2024_is_closed(calendar: NYSECalendar) -> None:
    """2024-11-28 (US Thanksgiving) is an NYSE holiday → False."""
    ts = _et(2024, 11, 28, 11, 0, 0)
    assert calendar.is_market_open(ts) is False


def test_black_friday_2024_half_day_after_close(calendar: NYSECalendar) -> None:
    """2024-11-29 at 13:01 ET is past the half-day close (13:00 ET) → False."""
    ts = _et(2024, 11, 29, 13, 1, 0)
    assert calendar.is_market_open(ts) is False


def test_normal_trading_hours(calendar: NYSECalendar) -> None:
    """2024-10-23 10:00 ET is a normal trading day and time → True."""
    ts = _et(2024, 10, 23, 10, 0, 0)
    assert calendar.is_market_open(ts) is True


def test_saturday_is_closed(calendar: NYSECalendar) -> None:
    """A Saturday at 11:00 ET is not a trading day → False."""
    # 2024-10-26 is a Saturday.
    ts = _et(2024, 10, 26, 11, 0, 0)
    assert calendar.is_market_open(ts) is False


def test_just_before_close_is_open(calendar: NYSECalendar) -> None:
    """15:59:59 ET on a normal trading day is within market hours → True."""
    ts = _et(2024, 10, 23, 15, 59, 59)
    assert calendar.is_market_open(ts) is True


def test_just_after_close_is_closed(calendar: NYSECalendar) -> None:
    """16:00:01 ET on a normal trading day is outside market hours → False."""
    ts = _et(2024, 10, 23, 16, 0, 1)
    assert calendar.is_market_open(ts) is False


def test_before_open_is_closed(calendar: NYSECalendar) -> None:
    """09:29:59 ET is before market open → False."""
    ts = _et(2024, 10, 23, 9, 29, 59)
    assert calendar.is_market_open(ts) is False


def test_at_open_is_open(calendar: NYSECalendar) -> None:
    """09:30:00 ET on a normal trading day is exactly at open → True."""
    ts = _et(2024, 10, 23, 9, 30, 0)
    assert calendar.is_market_open(ts) is True


def test_at_close_is_closed(calendar: NYSECalendar) -> None:
    """16:00:00 ET is at the close boundary → False (half-open interval [open, close))."""
    ts = _et(2024, 10, 23, 16, 0, 0)
    assert calendar.is_market_open(ts) is False


def test_next_open_from_weekend(calendar: NYSECalendar) -> None:
    """next_open from Saturday returns the following Monday open as UTC."""
    # 2024-10-26 Saturday 12:00 ET → next open is Mon 2024-10-28 09:30 ET.
    ts = _et(2024, 10, 26, 12, 0, 0)
    nxt = calendar.next_open(ts)
    nxt_et = nxt.astimezone(_ET)
    assert nxt_et.year == 2024
    assert nxt_et.month == 10
    assert nxt_et.day == 28
    assert nxt_et.hour == 9
    assert nxt_et.minute == 30


def test_next_open_from_after_close(calendar: NYSECalendar) -> None:
    """next_open from after Friday close returns the following Monday open."""
    # 2024-10-25 is a Friday; after 16:00 ET the next open is Mon 2024-10-28.
    ts = _et(2024, 10, 25, 17, 0, 0)
    nxt = calendar.next_open(ts)
    nxt_et = nxt.astimezone(_ET)
    assert nxt_et.day == 28
    assert nxt_et.hour == 9
    assert nxt_et.minute == 30


def test_market_calendar_gating(calendar: NYSECalendar) -> None:
    """Consolidated gating test matching FIRM-8 acceptance criteria.

    This is the mandatory test from the SPEC; the individual sub-tests above
    provide granular diagnostics when something fails.
    """
    # 1. NYSE holiday (Thanksgiving 2024) → closed
    assert calendar.is_market_open(_et(2024, 11, 28, 11, 0)) is False

    # 2. Day after Thanksgiving half-day: 13:01 ET → closed (past 13:00 early close)
    assert calendar.is_market_open(_et(2024, 11, 29, 13, 1)) is False

    # 3. Normal trading day/time: Oct 23 2024 10:00 ET → open
    assert calendar.is_market_open(_et(2024, 10, 23, 10, 0)) is True

    # 4. Saturday → closed
    assert calendar.is_market_open(_et(2024, 10, 26, 11, 0)) is False

    # 5. 15:59:59 ET → open; 16:00:01 ET → closed
    assert calendar.is_market_open(_et(2024, 10, 23, 15, 59, 59)) is True
    assert calendar.is_market_open(_et(2024, 10, 23, 16, 0, 1)) is False
