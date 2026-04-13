from __future__ import annotations

from typing import Any

import httpx

from .base import BaseIntegration


class AlphaVantageIntegration(BaseIntegration):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the Alpha Vantage integration client."""
        super().__init__(api_key=api_key, base_url=base_url or "https://www.alphavantage.co")

    def parse(self, response: httpx.Response) -> Any:
        """Parse an Alpha Vantage HTTP response into structured data."""
        raise NotImplementedError

    async def get_company_overview(self, symbol: str) -> Any:
        """Fetch company overview data for a ticker symbol."""
        raise NotImplementedError

    async def get_time_series(self, symbol: str, interval: str = "daily") -> Any:
        """Fetch time series market data for a ticker symbol."""
        raise NotImplementedError
