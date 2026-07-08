"""FRED macro series. Ported from PERIS src/integrations/fred.py (sync, schema adapted).

Two paths:
- Keyed JSON API (FRED_API_KEY) — the PERIS client, ported.
- Keyless fredgraph.csv fallback so the pipeline runs without any keys.

Macro series values are lagged one business day downstream (features/macro_regime.py)
because same-day FRED postings can land after the US close — point-in-time safety.
"""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pandas as pd

from ingestion.base import BaseClient

# Series used by the macro-regime feature family and risk gates.
MACRO_SERIES = {
    "VIX": "VIXCLS",  # CBOE VIX close
    "VIX3M": "VXVCLS",  # CBOE 3-month VIX close (term structure gate)
    "T10Y2Y": "T10Y2Y",  # 10y-2y treasury spread
    "DGS10": "DGS10",  # 10y treasury yield
    "FEDFUNDS": "DFF",  # effective fed funds (daily)
    "HY_OAS": "BAMLH0A0HYM2",  # high-yield credit spread
}


class FredClient(BaseClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or os.getenv("FRED_API_KEY", ""),
            base_url="https://api.stlouisfed.org",
        )

    @property
    def rate_limit_delay(self) -> float:
        return 0.5

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    def series_observations(
        self, series_id: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Daily observations as DataFrame(date, value). Uses keyed API if available."""
        if self.api_key:
            return self._series_api(series_id, start, end)
        return self._series_csv(series_id, start, end)

    def _series_api(self, series_id: str, start: str | None, end: str | None) -> pd.DataFrame:
        params: dict[str, Any] = {
            "api_key": self.api_key,
            "file_type": "json",
            "series_id": series_id,
            "sort_order": "asc",
            "limit": 100000,
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        data = self.fetch("/fred/series/observations", params=params)
        rows = [
            {"date": o["date"], "value": float(o["value"])}
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _series_csv(self, series_id: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Keyless fallback via fredgraph.csv."""
        self._throttle()
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def macro_panel(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Long panel: date, series, value for all tracked macro series."""
        frames = []
        for name, sid in MACRO_SERIES.items():
            df = self.series_observations(sid, start, end)
            df["series"] = name
            frames.append(df)
        return pd.concat(frames, ignore_index=True)[["date", "series", "value"]]
