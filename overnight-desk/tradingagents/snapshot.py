"""Point-in-time market snapshot: every number an agent is allowed to see.

One snapshot = four formatted text blocks (technical / fundamental / macro /
market-mood), all computed in pandas from the lake AS OF the decision session:
prices through t, macro through t-1 (FRED publication-lag convention, matching
features.macro_regime), fundamentals filed strictly before t, announcement
reactions on sessions <= t. The LLM formats and reasons; it never computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.fundamentals import (
    REVENUE_TAGS,
    _instant_series,
    _quarterly_flows,
    _ttm,
    _yoy,
)
from features.pead import earnings_events

MIN_HISTORY = 60

MACRO_LABELS = {
    "VIX": "VIX (implied vol)",
    "VIX3M": "VIX3M (3-month implied vol)",
    "DGS10": "10y Treasury yield %",
    "T10Y2Y": "2s10s curve slope %",
    "FEDFUNDS": "Fed funds rate %",
    "HY_OAS": "High-yield credit spread %",
}


@dataclass
class Snapshot:
    ticker: str
    asof: pd.Timestamp
    technical: str
    fundamental: str
    macro: str
    mood: str
    numbers: dict = field(default_factory=dict)  # raw values, for tests/telemetry


def _pct(x: float | None) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.1f}%"


def _rsi(closes: pd.Series, window: int = 14) -> float:
    delta = closes.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def technical_block(px: pd.Series, volume: pd.Series) -> tuple[str, dict]:
    ret = px.pct_change()
    n: dict[str, float] = {
        "r5": float(px.iloc[-1] / px.iloc[-6] - 1) if len(px) > 6 else np.nan,
        "r21": float(px.iloc[-1] / px.iloc[-22] - 1) if len(px) > 22 else np.nan,
        "r63": float(px.iloc[-1] / px.iloc[-64] - 1) if len(px) > 64 else np.nan,
        "r126": float(px.iloc[-1] / px.iloc[-127] - 1) if len(px) > 127 else np.nan,
        "vol21_ann": float(ret.iloc[-21:].std() * np.sqrt(252)),
        "rsi14": _rsi(px),
        "drawdown_252": float(px.iloc[-1] / px.iloc[-252:].max() - 1),
    }
    sma50 = px.iloc[-50:].mean()
    sma200 = px.iloc[-200:].mean() if len(px) >= 200 else np.nan
    n["vs_sma50"] = float(px.iloc[-1] / sma50 - 1)
    n["vs_sma200"] = float(px.iloc[-1] / sma200 - 1) if np.isfinite(sma200) else np.nan
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    n["macd_hist_pct"] = float((macd.iloc[-1] - signal.iloc[-1]) / px.iloc[-1])
    v63 = volume.iloc[-63:]
    n["volume_z"] = (
        float((volume.iloc[-5:].mean() - v63.mean()) / v63.std()) if v63.std() > 0 else 0.0
    )

    text = (
        f"Returns: 5d {_pct(n['r5'])}, 21d {_pct(n['r21'])}, 63d {_pct(n['r63'])}, "
        f"126d {_pct(n['r126'])}\n"
        f"Realized vol (21d, annualized): {_pct(n['vol21_ann'])}\n"
        f"RSI(14): {n['rsi14']:.0f}\n"
        f"Price vs SMA50: {_pct(n['vs_sma50'])}; vs SMA200: {_pct(n['vs_sma200'])}\n"
        f"MACD histogram (as % of price): {_pct(n['macd_hist_pct'])}\n"
        f"Drawdown from 252d high: {_pct(n['drawdown_252'])}\n"
        f"Volume, 5d avg vs 63d avg: {n['volume_z']:+.1f} standard deviations"
    )
    return text, n


def fundamental_block(facts: pd.DataFrame, asof: pd.Timestamp) -> tuple[str, dict]:
    facts = facts[facts["filed"] < asof]
    if facts.empty:
        return "No fundamental filings available.", {}
    n: dict[str, float] = {}
    lines: list[str] = []

    ni_ttm = _ttm(_quarterly_flows(facts, ["NetIncomeLoss"]))
    rev_ttm = _ttm(_quarterly_flows(facts, REVENUE_TAGS))
    cfo_ttm = _ttm(_quarterly_flows(facts, ["NetCashProvidedByUsedInOperatingActivities"]))
    equity = _instant_series(facts, "StockholdersEquity")
    assets = _instant_series(facts, "Assets")
    pfloat = _instant_series(facts, "EntityPublicFloat")

    def latest(series: pd.DataFrame) -> float | None:
        return float(series["val"].iloc[-1]) if len(series) else None

    ni, rev, cfo = latest(ni_ttm), latest(rev_ttm), latest(cfo_ttm)
    eq, at, fl = latest(equity), latest(assets), latest(pfloat)
    if rev:
        lines.append(f"TTM revenue: ${rev / 1e9:.1f}B")
        n["rev_ttm"] = rev
    if ni is not None:
        lines.append(f"TTM net income: ${ni / 1e9:.1f}B")
        n["ni_ttm"] = ni
    if cfo is not None:
        lines.append(f"TTM operating cash flow: ${cfo / 1e9:.1f}B")
    rev_yoy = _yoy(rev_ttm)
    ni_yoy = _yoy(ni_ttm, sign_safe=True)
    if len(rev_yoy):
        n["rev_yoy"] = float(rev_yoy["val"].iloc[-1])
        lines.append(f"Revenue growth (TTM yoy): {_pct(n['rev_yoy'])}")
    if len(ni_yoy):
        n["ni_growth"] = float(ni_yoy["val"].iloc[-1])
        lines.append(f"Net income growth (TTM yoy): {_pct(n['ni_growth'])}")
    if ni is not None and eq and eq > 0:
        n["roe"] = ni / eq
        lines.append(f"Return on equity (TTM): {_pct(n['roe'])}")
    if ni is not None and fl and fl > 0:
        n["earnings_yield"] = ni / fl
        lines.append(f"Earnings yield vs public float: {_pct(n['earnings_yield'])}")
    if ni is not None and cfo is not None and at and at > 0:
        n["accruals"] = (ni - cfo) / at
        lines.append(f"Accruals (NI-CFO)/assets: {_pct(n['accruals'])}")
    days = (asof - facts["filed"].max()).days
    n["days_since_filing"] = days
    lines.append(f"Most recent filing: {days} calendar days ago")
    return "\n".join(lines), n


def macro_block(macro: pd.DataFrame | None, asof: pd.Timestamp) -> tuple[str, dict]:
    if macro is None or macro.empty:
        return "No macro data available.", {}
    m = macro[macro["date"] < asof]  # published with a lag: strictly before t
    if m.empty:
        return "No macro data available.", {}
    n: dict[str, float] = {}
    lines: list[str] = []
    for series, label in MACRO_LABELS.items():
        s = m[m["series"] == series].sort_values("date")["value"]
        if s.empty:
            continue
        last = float(s.iloc[-1])
        prev = float(s.iloc[-22]) if len(s) > 22 else np.nan
        n[series] = last
        chg = f" (21d change {last - prev:+.2f})" if np.isfinite(prev) else ""
        lines.append(f"{label}: {last:.2f}{chg}")
    if "VIX" in n and "VIX3M" in n and n["VIX3M"] > 0:
        n["vix_term"] = n["VIX"] / n["VIX3M"]
        state = "INVERTED (stress)" if n["vix_term"] > 1 else "normal contango"
        lines.append(f"VIX term structure VIX/VIX3M: {n['vix_term']:.2f} — {state}")
    return "\n".join(lines), n


def mood_block(
    px: pd.Series,
    bench_px: pd.Series | None,
    events: pd.DataFrame | None,
    ticker: str,
    asof: pd.Timestamp,
) -> tuple[str, dict]:
    """Market-mood proxies: the market's own reactions stand in for social
    sentiment (no keyless social feed exists — see package docstring)."""
    n: dict[str, float] = {}
    lines: list[str] = []
    ret = px.pct_change()
    if bench_px is not None and len(bench_px) > 22:
        bret = bench_px.pct_change()
        n["rel_5d"] = float((1 + ret.iloc[-5:]).prod() - (1 + bret.iloc[-5:]).prod())
        n["rel_21d"] = float((1 + ret.iloc[-21:]).prod() - (1 + bret.iloc[-21:]).prod())
        lines.append(f"Relative to SPY: 5d {_pct(n['rel_5d'])}, 21d {_pct(n['rel_21d'])}")
    n["up_share_21d"] = float((ret.iloc[-21:] > 0).mean())
    lines.append(f"Share of up days, last 21 sessions: {n['up_share_21d'] * 100:.0f}%")

    if events is not None and not events.empty:
        ann = earnings_events(events)
        ann = ann[(ann["ticker"] == ticker) & (ann["filed"] <= asof)]
        if not ann.empty:
            last_ev = ann.iloc[-1]
            sessions = px.index
            accepted = last_ev["accepted"]
            if pd.isna(accepted):
                accepted = last_ev["filed"] + pd.Timedelta(hours=17)
            day = accepted.normalize() + (
                pd.Timedelta(0)
                if accepted - accepted.normalize() < pd.Timedelta(hours=9, minutes=30)
                else pd.Timedelta(days=1)
            )
            pos = sessions.searchsorted(day, side="left")
            if pos < len(sessions):
                reaction = sessions[pos]
                abn = float(ret.loc[reaction])
                if bench_px is not None and reaction in bench_px.index:
                    abn -= float(bench_px.pct_change().loc[reaction])
                n["last_earnings_reaction"] = abn
                n["sessions_since_earnings"] = int(len(sessions) - 1 - pos)
                lines.append(
                    f"Last earnings announcement: {n['sessions_since_earnings']} sessions ago, "
                    f"market reaction (abnormal return vs SPY) {_pct(abn)}"
                )
    if "last_earnings_reaction" not in n:
        lines.append("No earnings announcement observed in the window.")
    return "\n".join(lines), n


def build_snapshot(
    ticker: str,
    asof: pd.Timestamp | str,
    panel: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    benchmark: str = "SPY",
    intraday: dict | None = None,
    intraday_reason: str = "",
) -> Snapshot:
    """`intraday`: optional live-quote dict from tradingagents.intraday — appended
    to the technical block (and summarized in mood) as a clearly-labelled
    unofficial update; the PIT blocks above it are untouched."""
    asof = pd.Timestamp(asof)
    rows = panel[(panel["ticker"] == ticker) & (panel["date"] <= asof)].sort_values("date")
    if len(rows) < MIN_HISTORY:
        raise ValueError(f"{ticker}: only {len(rows)} sessions of history before {asof.date()}")
    px = rows.set_index("date")["close"]
    if px.index[-1] != asof:
        raise ValueError(f"{ticker}: {asof.date()} is not a session in the panel")
    volume = rows.set_index("date")["volume"]
    bench_rows = panel[(panel["ticker"] == benchmark) & (panel["date"] <= asof)]
    bench_px = bench_rows.set_index("date")["close"].sort_index() if len(bench_rows) else None

    tech_text, tech_n = technical_block(px, volume)
    fund_text, fund_n = (
        fundamental_block(fundamentals[fundamentals["ticker"] == ticker], asof)
        if fundamentals is not None
        else ("No fundamental filings available.", {})
    )
    macro_text, macro_n = macro_block(macro, asof)
    mood_text, mood_n = mood_block(px, bench_px, events, ticker, asof)
    if intraday:
        from tradingagents.intraday import format_intraday_block

        block = format_intraday_block(intraday, reason=intraday_reason)
        tech_text += "\n\n" + block
        mood_text += f"\n\nIntraday flash: stock is {intraday['change_pct']:+.1f}% today (live)."
        tech_n["intraday_change_pct"] = intraday["change_pct"]
    return Snapshot(
        ticker=ticker,
        asof=asof,
        technical=tech_text,
        fundamental=fund_text,
        macro=macro_text,
        mood=mood_text,
        numbers={**tech_n, **fund_n, **macro_n, **mood_n},
    )
