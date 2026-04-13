from __future__ import annotations

from typing import Any

import httpx

from .base import BaseIntegration


class GoogleTrendsIntegration(BaseIntegration):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the Google Trends integration client."""
        super().__init__(api_key=api_key, base_url=base_url or "https://trends.google.com")

    def parse(self, response: httpx.Response) -> Any:
        """Parse a Google Trends HTTP response into structured data."""
        raise NotImplementedError

    async def get_interest_over_time(
        self,
        keywords: list[str],
        timeframe: str = "today 12-m",
    ) -> Any:
        """Fetch trend interest-over-time data for one or more keywords."""
        raise NotImplementedError

    async def get_related_queries(self, keyword: str, timeframe: str = "today 12-m") -> Any:
        """Fetch related queries for a keyword."""
        raise NotImplementedError
