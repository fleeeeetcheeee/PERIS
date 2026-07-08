"""Estimate-free post-earnings-announcement-drift features (wave 6, EXPERIMENTAL).

The free tier has no analyst estimates, so the classic SUE is replaced by the
market's own verdict: the announcement-day abnormal return (close-to-close return
minus SPY) on the REACTION session. Announcement events are 8-K filings with item
2.02 from EDGAR submissions; a 10-Q/10-K acts as the event only when no 2.02 8-K
was filed in the 45 days before it or 7 days after (results are announced by 8-K
first and the report follows weeks later — treating those 10-Qs as events put 794
spurious clock-resets into a 2787-event stream, vs 3 genuine no-8-K fallbacks).

Reaction session: acceptance before 09:30 ET → the same session; otherwise the
next session (earnings 8-Ks overwhelmingly land after the close).

Point-in-time contract: a feature at date t uses only events whose reaction
session is <= t; the surprise uses closes through the reaction session only.

Features (prefix pead_), NaN when no event within DRIFT_SESSIONS or for ETFs:
- pead_surprise        abnormal return on the latest reaction session
- pead_days_since      sessions since that reaction session (0 on the day itself)
- pead_surprise_decay  surprise * 0.5 ** (days_since / 21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import paths
from features.common import validate_panel

FILING_EVENTS_CURATED = paths.CURATED / "filing_events.parquet"

DRIFT_SESSIONS = 63
DECAY_HALFLIFE = 21
FALLBACK_PRE_DAYS = 45  # an 8-K 2.02 this far before a 10-Q/K already announced it
FALLBACK_POST_DAYS = 7
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)


def earnings_events(filing_events: pd.DataFrame) -> pd.DataFrame:
    """One row per announcement: (ticker, filed, accepted). 8-K item 2.02 primary;
    a 10-Q/10-K is an event only when no 2.02 filing sits in
    [filed - FALLBACK_PRE_DAYS, filed + FALLBACK_POST_DAYS]."""
    ev = filing_events.copy()
    ev["filed"] = pd.to_datetime(ev["filed"])
    ev["accepted"] = pd.to_datetime(ev["accepted"])
    is_202 = (ev["form"] == "8-K") & ev["items"].fillna("").str.contains("2.02", regex=False)
    primary = ev[is_202]
    reports = ev[ev["form"].isin(("10-Q", "10-K"))]

    keep = [primary[["ticker", "filed", "accepted"]]]
    for ticker, grp in reports.groupby("ticker"):
        anchor = primary[primary["ticker"] == ticker]["filed"]

        def announced(d, a=anchor):
            return (
                (a >= d - pd.Timedelta(days=FALLBACK_PRE_DAYS))
                & (a <= d + pd.Timedelta(days=FALLBACK_POST_DAYS))
            ).any()

        gap = grp["filed"].apply(announced)
        keep.append(grp[~gap][["ticker", "filed", "accepted"]])
    out = pd.concat(keep, ignore_index=True)
    return out.sort_values(["ticker", "filed"]).reset_index(drop=True)


def _reaction_positions(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Index into `sessions` of each event's reaction session (len(sessions) = none)."""
    accepted = events["accepted"].fillna(events["filed"] + pd.Timedelta(hours=17))
    same_day = accepted.dt.normalize() + (accepted - accepted.dt.normalize() < MARKET_OPEN).map(
        {True: pd.Timedelta(0), False: pd.Timedelta(days=1)}
    )
    return sessions.searchsorted(same_day.to_numpy(), side="left")


def compute(
    panel: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    if events is None:
        if not FILING_EVENTS_CURATED.exists():
            raise FileNotFoundError(
                f"{FILING_EVENTS_CURATED} not found — run `uv run python -m ingestion.fundamentals`"
            )
        events = pd.read_parquet(FILING_EVENTS_CURATED)
    ann = earnings_events(events)

    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    ret = close.pct_change()
    bench_ret = ret[benchmark] if benchmark in ret.columns else pd.Series(0.0, index=ret.index)

    for c in ("pead_surprise", "pead_days_since", "pead_surprise_decay"):
        out[c] = np.nan

    for ticker, grp in ann.groupby("ticker"):
        if ticker not in ret.columns:
            continue
        mask = out["ticker"] == ticker
        sessions = pd.DatetimeIndex(out.loc[mask, "date"])
        n = len(sessions)
        surprise = np.full(n, np.nan)
        days_since = np.full(n, np.nan)

        tret = ret[ticker].reindex(sessions)
        bret = bench_ret.reindex(sessions)
        # chronological order: a later announcement overwrites an earlier one's tail
        for pos in np.sort(_reaction_positions(grp, sessions)):
            if pos >= n:
                continue
            abn = tret.iloc[pos] - bret.iloc[pos]
            if np.isnan(abn):
                continue
            end = min(pos + DRIFT_SESSIONS + 1, n)
            surprise[pos:end] = abn
            days_since[pos:end] = np.arange(end - pos)

        out.loc[mask, "pead_surprise"] = surprise
        out.loc[mask, "pead_days_since"] = days_since
        out.loc[mask, "pead_surprise_decay"] = surprise * 0.5 ** (days_since / DECAY_HALFLIFE)
    return out
