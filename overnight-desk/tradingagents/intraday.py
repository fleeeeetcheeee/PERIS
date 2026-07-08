"""Intraday quotes for live watch mode — keyless via yfinance.

During market hours yfinance's daily bars include TODAY as a live partial bar
(delayed ~15 min). That is deliberately NOT written to the lake: the lake is the
official nightly record; these quotes exist only to (a) trigger an intraday
review when a watchlist name moves hard and (b) be shown to the agents as a
clearly-labelled unofficial update on top of the point-in-time snapshot.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """{ticker: {last, prev_close, change_pct, day_high, day_low, from_high_pct,
    volume, prev_volume}} — empty dict on any vendor failure (watch skips a beat,
    never crashes the worker)."""
    import yfinance as yf

    try:
        df = yf.download(
            list(tickers),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:
        logger.warning("intraday fetch failed: %s", exc)
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, dict] = {}
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(subset=["Close"])
            if len(sub) < 2:
                continue
            today, prev = sub.iloc[-1], sub.iloc[-2]
            last, prev_close = float(today["Close"]), float(prev["Close"])
            day_high = float(today["High"])
            out[t] = {
                "last": last,
                "prev_close": prev_close,
                "change_pct": (last / prev_close - 1) * 100,
                "day_high": day_high,
                "day_low": float(today["Low"]),
                "from_high_pct": (last / day_high - 1) * 100 if day_high > 0 else 0.0,
                "volume": float(today["Volume"]),
                "prev_volume": float(prev["Volume"]),
            }
        except Exception:  # one bad ticker must not sink the rest
            logger.warning("intraday: no usable quote for %s", t)
    return out


def format_intraday_block(q: dict, reason: str = "") -> str:
    vol_ratio = q["volume"] / q["prev_volume"] if q.get("prev_volume") else None
    lines = [
        "INTRADAY UPDATE (delayed live quote — unofficial, not in the point-in-time "
        "record above; treat as breaking information):",
        f"Price now vs yesterday's close: {q['change_pct']:+.1f}%",
        f"Off today's high: {q['from_high_pct']:+.1f}%",
    ]
    if vol_ratio is not None:
        lines.append(f"Volume so far vs yesterday's full day: {vol_ratio:.0%}")
    if reason:
        lines.append(f"Review triggered because: {reason}")
    return "\n".join(lines)
