"""NYSE trading calendar helpers. All timestamps are America/New_York."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NY = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _xnys() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def sessions(start: date, end: date) -> pd.DatetimeIndex:
    """Trading sessions in [start, end], tz-naive dates in exchange time."""
    return _xnys().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def is_session(d: date) -> bool:
    return _xnys().is_session(pd.Timestamp(d))


def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent session whose close has passed, in exchange time."""
    now = now if now is not None else pd.Timestamp.now(tz=NY)
    if now.tzinfo is None:
        now = now.tz_localize(NY)
    cal = _xnys()
    session = cal.minute_to_past_session(now.tz_convert("UTC"))
    return session


def is_open_now(now: pd.Timestamp | None = None) -> bool:
    """True while the NYSE regular session is trading."""
    now = now if now is not None else pd.Timestamp.now(tz=NY)
    if now.tzinfo is None:
        now = now.tz_localize(NY)
    cal = _xnys()
    minute = now.tz_convert("UTC").floor("min")
    try:
        return bool(cal.is_open_on_minute(minute))
    except ValueError:  # outside the calendar's minute range
        return False


def next_sessions(after: date, n: int) -> pd.DatetimeIndex:
    """The next n sessions strictly after `after`."""
    cal = _xnys()
    start = pd.Timestamp(after) + pd.Timedelta(days=1)
    end = pd.Timestamp(after) + pd.Timedelta(days=30 + 2 * n)
    all_sessions = cal.sessions_in_range(start, end)
    return all_sessions[:n]


def shift_session(d: pd.Timestamp, offset: int) -> pd.Timestamp:
    """Shift a session by `offset` trading days (can be negative)."""
    cal = _xnys()
    return cal.session_offset(pd.Timestamp(d).tz_localize(None), offset)
