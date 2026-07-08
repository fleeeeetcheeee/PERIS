"""Tiingo — primary EOD source when TIINGO_API_KEY is set. Free tier; all pulls cached."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from ingestion.base import BaseClient


class TiingoClient(BaseClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or os.getenv("TIINGO_API_KEY", ""),
            base_url="https://api.tiingo.com",
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def rate_limit_delay(self) -> float:
        return 1.2  # free tier: stay well under hourly caps

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}", "Accept": "application/json"}

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    def daily_bars(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        params: dict[str, str] = {"format": "json"}
        if start:
            params["startDate"] = start
        if end:
            params["endDate"] = end
        rows = self.fetch(f"/tiingo/daily/{ticker.lower()}/prices", params=params)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "ticker"]
            )
        # Use adjusted fields so splits/dividends don't fake returns. Select before
        # renaming — the response also carries unadjusted open/high/low/close/volume,
        # and renaming in place would create duplicate column names.
        df = df[["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]].rename(
            columns={
                "adjOpen": "open",
                "adjHigh": "high",
                "adjLow": "low",
                "adjClose": "close",
                "adjVolume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = ticker.upper()
        return df[["date", "open", "high", "low", "close", "volume", "ticker"]]
