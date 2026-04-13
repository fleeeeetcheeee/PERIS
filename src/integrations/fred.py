from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseIntegration

# Key macroeconomic series tracked by PERIS
MACRO_SERIES = {
    "GDP": "GDP",                          # Gross Domestic Product
    "CPI": "CPIAUCSL",                     # Consumer Price Index
    "FED_FUNDS_RATE": "FEDFUNDS",          # Federal Funds Rate
    "UNEMPLOYMENT": "UNRATE",             # Unemployment Rate
    "10Y_TREASURY": "GS10",               # 10-Year Treasury Yield
    "INDUSTRIAL_PRODUCTION": "INDPRO",    # Industrial Production Index
    "RETAIL_SALES": "RSXFS",              # Advance Retail Sales
    "SECTOR_EMPLOYMENT_MFG": "MANEMP",    # Manufacturing Employment
    "SECTOR_EMPLOYMENT_INFO": "USINFO",   # Information Sector Employment
    "SECTOR_EMPLOYMENT_FINANCE": "USFIRE", # Finance/Insurance Employment
}


class FredIntegration(BaseIntegration):
    """Async client for the FRED (Federal Reserve Economic Data) API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(
            api_key=api_key or os.getenv("FRED_API_KEY", ""),
            base_url=base_url or "https://api.stlouisfed.org",
        )

    @property
    def rate_limit_delay(self) -> float:
        return 0.1

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    def _base_params(self) -> dict[str, str]:
        return {"api_key": self.api_key or "", "file_type": "json"}

    # ------------------------------------------------------------------
    # Low-level series endpoints
    # ------------------------------------------------------------------

    async def get_series(self, series_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch metadata for a FRED series."""
        merged = {**self._base_params(), "series_id": series_id, **(params or {})}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.base_url}/fred/series", params=merged)
            resp.raise_for_status()
            return resp.json()

    async def get_series_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch observations for a FRED series, newest-first."""
        params: dict[str, Any] = {
            **self._base_params(),
            "series_id": series_id,
            "sort_order": "desc",
            "limit": limit,
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.base_url}/fred/series/observations", params=params)
            resp.raise_for_status()
            data = resp.json()

        return [
            {"date": obs["date"], "value": self._parse_value(obs["value"])}
            for obs in data.get("observations", [])
            if obs.get("value") != "."
        ]

    def _parse_value(self, raw: str) -> float | None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    async def get_latest_value(self, series_id: str) -> dict[str, Any]:
        """Return the most recent (date, value) pair for a series."""
        obs = await self.get_series_observations(series_id, limit=1)
        if obs:
            return obs[0]
        return {"date": None, "value": None}

    async def fetch_macro_snapshot(self) -> dict[str, dict[str, Any]]:
        """
        Fetch the latest value for all tracked macro series.
        Returns a dict keyed by friendly name, e.g. {"GDP": {"date": "...", "value": 28000.0}}.
        """
        import asyncio

        async def _fetch(name: str, sid: str) -> tuple[str, dict[str, Any]]:
            val = await self.get_latest_value(sid)
            return name, val

        results = await asyncio.gather(
            *[_fetch(name, sid) for name, sid in MACRO_SERIES.items()],
            return_exceptions=True,
        )

        snapshot: dict[str, dict[str, Any]] = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            name, val = item
            snapshot[name] = val

        return snapshot

    async def get_gdp(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self.get_series_observations("GDP", limit=limit)

    async def get_cpi(self, limit: int = 12) -> list[dict[str, Any]]:
        return await self.get_series_observations("CPIAUCSL", limit=limit)

    async def get_interest_rates(self, limit: int = 12) -> list[dict[str, Any]]:
        return await self.get_series_observations("FEDFUNDS", limit=limit)

    async def get_sector_employment(
        self,
        sector: str = "MFG",
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Fetch employment for a sector key defined in MACRO_SERIES."""
        series_map = {
            "MFG": "MANEMP",
            "INFO": "USINFO",
            "FINANCE": "USFIRE",
        }
        sid = series_map.get(sector.upper(), "MANEMP")
        return await self.get_series_observations(sid, limit=limit)
