"""Stooq bulk historical backfill. Keyless; used for one-time backfill into data/raw/.

Stooq daily bars are split-adjusted but NOT dividend-adjusted. Total-return features
built on these closes understate dividend payers slightly; flagged per hard-constraint 2
(if unsure about point-in-time safety, flag it). Tiingo adjClose supersedes these rows
when a TIINGO_API_KEY is configured.
"""

from __future__ import annotations

import io
import logging

import httpx
import pandas as pd

from ingestion.base import BaseClient

logger = logging.getLogger(__name__)


def to_stooq_symbol(ticker: str) -> str:
    # BRK.B -> brk-b.us
    return ticker.lower().replace(".", "-") + ".us"


class StooqClient(BaseClient):
    def __init__(self) -> None:
        super().__init__(base_url="https://stooq.com")

    @property
    def rate_limit_delay(self) -> float:
        return 0.6

    def headers(self) -> dict[str, str]:
        return {"User-Agent": "overnight-desk research (personal, non-commercial)"}

    def parse(self, response: httpx.Response) -> pd.DataFrame:
        text = response.text
        if not text or text.startswith("No data") or "Exceeded" in text[:200]:
            raise RuntimeError(f"stooq returned no data: {text[:100]!r}")
        df = pd.read_csv(io.StringIO(text))
        expected = {"Date", "Open", "High", "Low", "Close"}
        if not expected.issubset(df.columns):
            raise RuntimeError(f"unexpected stooq columns: {list(df.columns)}")
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]]

    def daily_bars(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        params: dict[str, str] = {"s": to_stooq_symbol(ticker), "i": "d"}
        if start:
            params["d1"] = start.replace("-", "")
        if end:
            params["d2"] = end.replace("-", "")
        df = self.fetch("/q/d/l/", params=params)
        df["ticker"] = ticker.upper()
        return df
