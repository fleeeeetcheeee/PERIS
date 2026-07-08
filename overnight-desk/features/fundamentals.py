"""Fundamental features from EDGAR companyfacts vintages (wave 6, EXPERIMENTAL).

Point-in-time contract: a fact filed on day f becomes usable at the NEXT session
(filings land at arbitrary times of day, so same-day use would peek). Every join
is as-of on `filed`, never on the fiscal period end; only the FIRST vintage of
each (tag, start, end) is used, so restatements can't rewrite history.

Valuation denominators use dei EntityPublicFloat (a DOLLAR value filed on each
10-K cover) — NOT close * shares outstanding: the panel's close is split-adjusted
to today's basis while EDGAR share counts are as-reported, so price*shares is
wrong by every FUTURE split factor (AAPL 2018 printed 5x its true market cap).
Public float is crude (annual cadence, excludes insiders) but PIT by construction.

Features (prefix fund_), all NaN before the first available filing and for ETFs:
- fund_ey           TTM net income / public float
- fund_cfo_yield    TTM operating cash flow / public float
- fund_roe          TTM net income / stockholders' equity
- fund_rev_yoy      TTM revenue vs TTM revenue four quarters earlier
- fund_ni_growth    TTM net income change / abs(prior), sign-safe
- fund_asset_growth total assets vs four quarters earlier
- fund_accruals     (TTM net income - TTM CFO) / total assets

Quarterly flows: Q4 is derived as FY minus the three quarters inside the fiscal
year (available only once ALL components are filed); TTM requires 4 quarters
spanning ~a year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import paths
from features.common import validate_panel

FUNDAMENTALS_CURATED = paths.CURATED / "fundamentals.parquet"

REVENUE_TAGS = [  # priority order — later filers switched to the ASC 606 tag
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)
TTM_SPAN_DAYS = (250, 290)  # first-to-last END of 4 consecutive quarters = ~3 quarters


def _first_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (start, end): the earliest-filed value."""
    return (
        df.sort_values("filed", kind="stable")
        .drop_duplicates(subset=["start", "end"], keep="first")
        .reset_index(drop=True)
    )


def _quarterly_flows(facts: pd.DataFrame, tag_priority: list[str]) -> pd.DataFrame:
    """Quarterly (end, filed, val) for a flow concept, from three extraction paths:

    1. direct 3-month duration rows (income-statement style),
    2. differences of same-start cumulative durations (cash-flow statements report
       YTD only — Q2 = 6moYTD − Q1, ... Q4 = FY − 9moYTD),
    3. annual minus its three inside quarters (Q4 for filers whose quarters carry
       their own start dates).

    `filed` on any derived value is the max of its components' filed dates — the
    value does not exist until every component is public.
    """
    rows = facts[facts["tag"].isin(tag_priority) & facts["start"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    rows["prio"] = rows["tag"].map({t: i for i, t in enumerate(tag_priority)})
    rows = rows.sort_values(["prio", "filed"], kind="stable").drop_duplicates(
        subset=["start", "end"], keep="first"
    )
    dur = (rows["end"] - rows["start"]).dt.days
    direct = rows[dur.between(*QUARTER_DAYS)][["start", "end", "filed", "val"]]
    annuals = rows[dur.between(*ANNUAL_DAYS)][["start", "end", "filed", "val"]]

    diffed = []
    for _, grp in rows.groupby("start"):
        grp = grp.sort_values("end")
        prev = None
        for r in grp.itertuples():
            if prev is not None and QUARTER_DAYS[0] <= (r.end - prev.end).days <= QUARTER_DAYS[1]:
                diffed.append(
                    {"end": r.end, "filed": max(r.filed, prev.filed), "val": r.val - prev.val}
                )
            prev = r

    q4 = []
    for a in annuals.itertuples():
        comp = direct[(direct["start"] >= a.start) & (direct["end"] < a.end)]
        if len(comp) != 3:
            continue
        q4.append(
            {
                "end": a.end,
                "filed": max(a.filed, comp["filed"].max()),
                "val": a.val - comp["val"].sum(),
            }
        )

    combined = pd.concat(
        [
            direct[["end", "filed", "val"]].assign(src=0),
            pd.DataFrame(diffed, columns=["end", "filed", "val"]).assign(src=1),
            pd.DataFrame(q4, columns=["end", "filed", "val"]).assign(src=2),
        ],
        ignore_index=True,
    )
    return (
        combined.sort_values(["src", "filed"], kind="stable")
        .drop_duplicates(subset=["end"], keep="first")[["end", "filed", "val"]]
        .sort_values("end")
        .reset_index(drop=True)
    )


def _ttm(quarterly: pd.DataFrame) -> pd.DataFrame:
    """(end, filed, val): rolling 4-quarter sum; filed = latest component filing.
    Four CONSECUTIVE quarter ends span ~9 months — windows with a missing quarter
    (stretched span) are rejected, not summed."""
    if len(quarterly) < 4:
        return pd.DataFrame(columns=["end", "filed", "val"])
    q = quarterly.sort_values("end").reset_index(drop=True)
    rows = []
    for i in range(3, len(q)):
        window = q.iloc[i - 3 : i + 1]
        span = (window["end"].iloc[-1] - window["end"].iloc[0]).days
        if not (TTM_SPAN_DAYS[0] <= span <= TTM_SPAN_DAYS[1]):
            continue
        rows.append(
            {
                "end": window["end"].iloc[-1],
                "filed": window["filed"].max(),
                "val": window["val"].sum(),
            }
        )
    return pd.DataFrame(rows, columns=["end", "filed", "val"])


def _instant_series(facts: pd.DataFrame, tag: str) -> pd.DataFrame:
    """(end, filed, val) for an instant concept, frontier-filtered: a comparative
    old-period value filed late may never overwrite a newer period."""
    rows = facts[(facts["tag"] == tag) & facts["start"].isna()]
    if rows.empty:
        rows = facts[facts["tag"] == tag]  # dei rows carry no start either way
    if rows.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    rows = (
        rows.sort_values("filed", kind="stable")
        .drop_duplicates(subset=["end"], keep="first")
        .sort_values("filed", kind="stable")
        .reset_index(drop=True)
    )
    frontier = rows["end"].cummax()
    return rows[rows["end"] >= frontier][["end", "filed", "val"]].reset_index(drop=True)


def _yoy(series: pd.DataFrame, sign_safe: bool = False) -> pd.DataFrame:
    """(end, filed, val=growth) vs the value 4 quarter-ends earlier."""
    if len(series) < 5:
        return pd.DataFrame(columns=["end", "filed", "val"])
    s = series.sort_values("end").reset_index(drop=True)
    prev = s["val"].shift(4)
    span = (s["end"] - s["end"].shift(4)).dt.days
    if sign_safe:
        growth = (s["val"] - prev) / prev.abs()
    else:
        growth = s["val"] / prev - 1.0
    growth = growth.where(span.between(300, 430))  # gaps in the quarter chain → no yoy
    out = pd.DataFrame({"end": s["end"], "filed": s["filed"], "val": growth})
    return out.dropna(subset=["val"]).reset_index(drop=True)


def _asof_sessions(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Value known at each session: latest event with filed + 1 session <= t,
    implemented as filed strictly < session date (calendar day granularity)."""
    if events.empty:
        return pd.Series(np.nan, index=sessions)
    ev = events.sort_values("filed", kind="stable")
    merged = pd.merge_asof(
        pd.DataFrame({"date": sessions}),
        ev.rename(columns={"filed": "date"})[["date", "val"]],
        on="date",
        allow_exact_matches=False,
    )
    return pd.Series(merged["val"].to_numpy(), index=sessions)


def compute(
    panel: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    if fundamentals is None:
        if not FUNDAMENTALS_CURATED.exists():
            raise FileNotFoundError(
                f"{FUNDAMENTALS_CURATED} not found — run `uv run python -m ingestion.fundamentals`"
            )
        fundamentals = pd.read_parquet(FUNDAMENTALS_CURATED)
    fundamentals = fundamentals.copy()
    for col in ("start", "end", "filed"):
        fundamentals[col] = pd.to_datetime(fundamentals[col])

    cols = [
        "fund_ey",
        "fund_cfo_yield",
        "fund_roe",
        "fund_rev_yoy",
        "fund_ni_growth",
        "fund_asset_growth",
        "fund_accruals",
    ]
    for c in cols:
        out[c] = np.nan

    for ticker, facts in fundamentals.groupby("ticker"):
        mask = out["ticker"] == ticker
        if not mask.any():
            continue
        sessions = pd.DatetimeIndex(out.loc[mask, "date"])

        ni_q = _quarterly_flows(facts, ["NetIncomeLoss"])
        cfo_q = _quarterly_flows(facts, ["NetCashProvidedByUsedInOperatingActivities"])
        rev_q = _quarterly_flows(facts, REVENUE_TAGS)
        ni_ttm_ev, cfo_ttm_ev, rev_ttm_ev = _ttm(ni_q), _ttm(cfo_q), _ttm(rev_q)
        equity_ev = _instant_series(facts, "StockholdersEquity")
        assets_ev = _instant_series(facts, "Assets")
        float_ev = _instant_series(facts, "EntityPublicFloat")

        ni_ttm = _asof_sessions(ni_ttm_ev, sessions)
        cfo_ttm = _asof_sessions(cfo_ttm_ev, sessions)
        equity = _asof_sessions(equity_ev, sessions)
        assets = _asof_sessions(assets_ev, sessions)
        pfloat = _asof_sessions(float_ev, sessions).where(lambda s: s > 0)

        out.loc[mask, "fund_ey"] = (ni_ttm / pfloat).to_numpy()
        out.loc[mask, "fund_cfo_yield"] = (cfo_ttm / pfloat).to_numpy()
        out.loc[mask, "fund_roe"] = (ni_ttm / equity.where(equity > 0)).to_numpy()
        out.loc[mask, "fund_rev_yoy"] = _asof_sessions(_yoy(rev_ttm_ev), sessions).to_numpy()
        out.loc[mask, "fund_ni_growth"] = _asof_sessions(
            _yoy(ni_ttm_ev, sign_safe=True), sessions
        ).to_numpy()
        out.loc[mask, "fund_asset_growth"] = _asof_sessions(_yoy(assets_ev), sessions).to_numpy()
        out.loc[mask, "fund_accruals"] = ((ni_ttm - cfo_ttm) / assets.where(assets > 0)).to_numpy()
    return out
