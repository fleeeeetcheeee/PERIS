"""Wave-6 feature tests: fundamentals + PEAD.

The lookahead tests here are the fundamentals/PEAD analogue of test_lookahead.py:
those corrupt future PRICES; these additionally corrupt future FILINGS — both the
values and the filed dates — because the leak vector for filing-based features is
the join key, not the price panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import fundamentals, pead
from features.fundamentals import _quarterly_flows, _ttm
from features.pead import earnings_events

CUTOFF = pd.Timestamp("2022-12-15")


def make_facts(ticker: str = "AAA") -> pd.DataFrame:
    """Synthetic companyfacts vintages: direct quarterly NI/revenue, YTD-cumulative
    CFO (cash-flow statements never report 3-month rows), quarterly instants, and
    an annual public float."""
    rows: list[dict] = []
    cfo_ytd = 0.0
    for i, p in enumerate(pd.period_range("2020Q1", "2023Q1", freq="Q")):
        s, e = p.start_time.normalize(), p.end_time.normalize()
        filed = e + pd.Timedelta(days=40)
        ni, rev, cfo_q = 100.0 + i, 1000.0 + 10 * i, 120.0 + i
        rows.append(dict(tag="NetIncomeLoss", start=s, end=e, filed=filed, val=ni))
        rows.append(dict(tag="Revenues", start=s, end=e, filed=filed, val=rev))
        year_start = pd.Timestamp(f"{p.year}-01-01")
        cfo_ytd = cfo_q if p.quarter == 1 else cfo_ytd + cfo_q
        rows.append(
            dict(
                tag="NetCashProvidedByUsedInOperatingActivities",
                start=year_start,
                end=e,
                filed=filed,
                val=cfo_ytd,
            )
        )
        rows.append(dict(tag="Assets", start=None, end=e, filed=filed, val=10_000.0 + 100 * i))
        rows.append(
            dict(tag="StockholdersEquity", start=None, end=e, filed=filed, val=5_000.0 + 50 * i)
        )
        if p.quarter == 4:
            rows.append(dict(tag="EntityPublicFloat", start=None, end=e, filed=filed, val=50_000.0))
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    df["unit"] = "USD"
    df["form"] = "10-Q"
    df["fp"] = ""
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    return df


def make_events(ticker: str = "AAA") -> pd.DataFrame:
    """8-K 2.02 announcements: quarterly, accepted after the close."""
    rows = []
    for filed in pd.date_range("2022-02-01", "2023-02-01", freq="91D"):
        filed = filed + pd.offsets.BDay(0)  # keep on a weekday
        rows.append(
            dict(
                ticker=ticker,
                form="8-K",
                filed=filed,
                accepted=filed + pd.Timedelta(hours=16, minutes=30),
                items="2.02,9.01",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def facts() -> pd.DataFrame:
    return make_facts()


@pytest.fixture
def events() -> pd.DataFrame:
    return make_events()


def _corrupt_future_prices(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = panel.copy()
    future = out["date"] > cutoff
    rng = np.random.default_rng(99)
    for col in ("open", "high", "low", "close", "volume"):
        out.loc[future, col] = out.loc[future, col].values * rng.uniform(0.5, 2.0, future.sum())
    return out


def _past(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("date", "ticker")]
    return df.loc[df["date"] <= CUTOFF, cols].reset_index(drop=True)


# ---------------------------------------------------------------- fundamentals


def test_fundamentals_ignore_future_filing_values(panel, facts):
    base = fundamentals.compute(panel, fundamentals=facts)
    corrupted = facts.copy()
    future = corrupted["filed"] > CUTOFF
    corrupted.loc[future, "val"] = corrupted.loc[future, "val"] * 7.0 + 123.0
    after = fundamentals.compute(panel, fundamentals=corrupted)
    pd.testing.assert_frame_equal(_past(base), _past(after), check_exact=True)


def test_fundamentals_ignore_future_filed_dates(panel, facts):
    base = fundamentals.compute(panel, fundamentals=facts)
    corrupted = facts.copy()
    future = corrupted["filed"] > CUTOFF
    corrupted.loc[future, "filed"] = corrupted.loc[future, "filed"] + pd.Timedelta(days=200)
    after = fundamentals.compute(panel, fundamentals=corrupted)
    pd.testing.assert_frame_equal(_past(base), _past(after), check_exact=True)


def test_fundamentals_ignore_future_prices(panel, facts):
    base = fundamentals.compute(panel, fundamentals=facts)
    after = fundamentals.compute(_corrupt_future_prices(panel, CUTOFF), fundamentals=facts)
    pd.testing.assert_frame_equal(_past(base), _past(after), check_exact=True)


def test_fundamentals_available_strictly_after_filed(panel, facts):
    """A fact filed on day F must not move features on any session <= F."""
    filed = pd.Timestamp("2022-06-01")
    extra = facts.iloc[[0]].copy()
    extra["tag"], extra["start"], extra["end"] = "Assets", None, pd.Timestamp("2022-04-30")
    extra["filed"], extra["val"] = filed, 99_999.0
    base = fundamentals.compute(panel, fundamentals=facts)
    after = fundamentals.compute(panel, fundamentals=pd.concat([facts, extra], ignore_index=True))
    changed = base.fillna(-1).ne(after.fillna(-1)).any(axis=1) & (base["ticker"] == "AAA")
    assert changed.any(), "the extra filing never showed up at all"
    assert base.loc[changed, "date"].min() > filed


def test_quarterly_flows_and_ttm_math(facts):
    """CFO quarterly values must come out of YTD differencing; TTM = last 4 quarters."""
    aaa = facts[facts["ticker"] == "AAA"]
    cfo_q = _quarterly_flows(aaa, ["NetCashProvidedByUsedInOperatingActivities"])
    # quarters were built as 120+i for i = 0..12
    assert np.allclose(cfo_q.sort_values("end")["val"].to_numpy(), 120.0 + np.arange(13))
    ni_ttm = _ttm(_quarterly_flows(aaa, ["NetIncomeLoss"]))
    last = ni_ttm.sort_values("end").iloc[-1]
    assert last["val"] == pytest.approx(sum(100.0 + i for i in range(9, 13)))
    # TTM availability = latest component filing, not the period end
    assert last["filed"] == pd.Timestamp("2023-03-31") + pd.Timedelta(days=40)


def test_ttm_rejects_gapped_windows(facts):
    aaa = facts[facts["ticker"] == "AAA"]
    q = _quarterly_flows(aaa, ["NetIncomeLoss"])
    gapped = q[q["end"] != pd.Timestamp("2022-06-30")]  # knock out one quarter
    ttm = _ttm(gapped)
    spans = ttm.sort_values("end")["end"].diff().dt.days.dropna()
    assert not ttm.empty
    # no TTM window may bridge the hole: every retained window is 4 consecutive quarters
    assert (
        (ttm["end"] < pd.Timestamp("2022-06-30")) | (ttm["end"] >= pd.Timestamp("2023-03-31"))
    ).all(), spans


# ------------------------------------------------------------------------ pead


def test_pead_ignore_future_prices_and_events(panel, events):
    base = pead.compute(panel, events=events)
    corrupted_events = events.copy()
    future = corrupted_events["filed"] > CUTOFF
    corrupted_events.loc[future, "filed"] = corrupted_events.loc[future, "filed"] + pd.Timedelta(
        days=90
    )
    corrupted_events.loc[future, "accepted"] = corrupted_events.loc[
        future, "accepted"
    ] + pd.Timedelta(days=90)
    after = pead.compute(_corrupt_future_prices(panel, CUTOFF), events=corrupted_events)
    pd.testing.assert_frame_equal(_past(base), _past(after), check_exact=True)


def test_pead_reaction_next_session_after_close(panel):
    filed = pd.Timestamp("2022-06-01")  # a Wednesday
    ev = pd.DataFrame(
        [
            dict(
                ticker="AAA",
                form="8-K",
                filed=filed,
                accepted=filed + pd.Timedelta(hours=16, minutes=30),
                items="2.02",
            )
        ]
    )
    out = pead.compute(panel, events=ev)
    aaa = out[out["ticker"] == "AAA"].set_index("date")
    assert np.isnan(aaa.loc[filed, "pead_surprise"])
    reaction = pd.Timestamp("2022-06-02")
    close = panel.pivot(index="date", columns="ticker", values="close")
    expected = close["AAA"].pct_change().loc[reaction] - close["SPY"].pct_change().loc[reaction]
    assert aaa.loc[reaction, "pead_surprise"] == pytest.approx(expected)
    assert aaa.loc[reaction, "pead_days_since"] == 0


def test_pead_reaction_same_session_before_open(panel):
    filed = pd.Timestamp("2022-06-01")
    ev = pd.DataFrame(
        [
            dict(
                ticker="AAA",
                form="8-K",
                filed=filed,
                accepted=filed + pd.Timedelta(hours=8),
                items="2.02",
            )
        ]
    )
    out = pead.compute(panel, events=ev)
    aaa = out[out["ticker"] == "AAA"].set_index("date")
    assert aaa.loc[filed, "pead_days_since"] == 0


def test_pead_drift_expires(panel):
    filed = pd.Timestamp("2022-03-01")
    ev = pd.DataFrame(
        [
            dict(
                ticker="AAA",
                form="8-K",
                filed=filed,
                accepted=filed + pd.Timedelta(hours=17),
                items="2.02",
            )
        ]
    )
    out = pead.compute(panel, events=ev)
    aaa = out[out["ticker"] == "AAA"].reset_index(drop=True)
    live = aaa["pead_days_since"].notna()
    assert live.sum() == pead.DRIFT_SESSIONS + 1
    assert aaa.loc[live, "pead_days_since"].max() == pead.DRIFT_SESSIONS


def test_pead_10q_fallback_only_without_8k(events):
    ev = pd.concat(
        [
            events,  # AAA has 8-Ks
            pd.DataFrame(
                [
                    # BBB files a 10-Q with no 2.02 8-K anywhere near it -> event
                    dict(
                        ticker="BBB",
                        form="10-Q",
                        filed=pd.Timestamp("2022-05-10"),
                        accepted=pd.Timestamp("2022-05-10 17:00"),
                        items="",
                    ),
                    # AAA files a 10-Q right next to an existing 8-K -> NOT an event
                    dict(
                        ticker="AAA",
                        form="10-Q",
                        filed=events["filed"].iloc[0] + pd.Timedelta(days=2),
                        accepted=events["filed"].iloc[0] + pd.Timedelta(days=2, hours=17),
                        items="",
                    ),
                    # AAA files the 10-Q a month AFTER the 8-K announcement — the
                    # results are already public, so this must NOT reset the clock
                    dict(
                        ticker="AAA",
                        form="10-Q",
                        filed=events["filed"].iloc[0] + pd.Timedelta(days=30),
                        accepted=events["filed"].iloc[0] + pd.Timedelta(days=30, hours=17),
                        items="",
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    ann = earnings_events(ev)
    assert (ann["ticker"] == "BBB").sum() == 1
    assert (ann["ticker"] == "AAA").sum() == len(events)
