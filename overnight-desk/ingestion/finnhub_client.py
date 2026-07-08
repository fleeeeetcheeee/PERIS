"""Finnhub — earnings calendar (free tier, 60 calls/min). Requires FINNHUB_API_KEY.

Without a key the earnings gate degrades to a no-op with a loud warning — it never
silently pretends names were screened.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from ingestion.base import BaseClient


class FinnhubClient(BaseClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or os.getenv("FINNHUB_API_KEY", ""),
            base_url="https://finnhub.io/api/v1",
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def rate_limit_delay(self) -> float:
        return 1.1  # 60 calls/min

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    def earnings_calendar(self, start: str, end: str) -> pd.DataFrame:
        """Confirmed earnings dates in [start, end]. Columns: ticker, earnings_date."""
        data = self.fetch(
            "/calendar/earnings",
            params={"from": start, "to": end, "token": self.api_key},
        )
        rows = data.get("earningsCalendar", [])
        if not rows:
            return pd.DataFrame(columns=["ticker", "earnings_date"])
        df = pd.DataFrame(rows)
        df = df.rename(columns={"symbol": "ticker", "date": "earnings_date"})
        df["ticker"] = df["ticker"].str.upper()
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df[["ticker", "earnings_date"]].drop_duplicates()
