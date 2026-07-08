"""Yahoo Finance via yfinance — FALLBACK ONLY, per CLAUDE2.md data-source policy.

Used when no TIINGO_API_KEY is configured and Stooq is unreachable (it now fronts its
CSV endpoint with a browser-verification challenge). auto_adjust=True returns
split- and dividend-adjusted OHLC.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooClient:
    available = True

    def daily_bars(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        # yfinance `end` is exclusive; push it one day so the last session is included.
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat() if end else None
        df = yf.download(
            ticker,
            start=start,
            end=end_excl,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "ticker"]
            )
        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = ticker.upper()
        return df[["date", "open", "high", "low", "close", "volume", "ticker"]]
