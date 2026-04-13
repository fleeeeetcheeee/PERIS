from __future__ import annotations

import asyncio
from typing import Any

import httpx
import yfinance as yf

from .base import BaseIntegration


class YahooFinanceIntegration(BaseIntegration):
    """Wrapper around yfinance for price, fundamentals, and EV/EBITDA data."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or "")

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    # ------------------------------------------------------------------
    # Single-ticker helpers (sync wrappers run in executor)
    # ------------------------------------------------------------------

    def _fetch_ticker_data(self, symbol: str) -> dict[str, Any]:
        """Synchronous fetch — called via run_in_executor."""
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName", symbol),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "enterprise_value": info.get("enterpriseValue"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "debt_to_equity": info.get("debtToEquity"),
            "employee_count": info.get("fullTimeEmployees"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }

    def _fetch_historical(self, symbol: str, period: str, interval: str) -> list[dict[str, Any]]:
        """Synchronous history fetch — called via run_in_executor."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            return []
        hist = hist.reset_index()
        return [
            {
                "date": str(row["Date"])[:10],
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
            for _, row in hist.iterrows()
        ]

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch latest quote and key fundamentals for a single ticker."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_ticker_data, symbol)

    async def get_historical_prices(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        """Fetch historical OHLCV data for a ticker."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_historical, symbol, period, interval
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    async def get_quotes_batch(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Fetch quotes for multiple tickers concurrently."""
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def get_ev_ebitda(self, symbol: str) -> dict[str, Any]:
        """Return EV and EBITDA multiples for a ticker."""
        data = await self.get_quote(symbol)
        return {
            "symbol": data["symbol"],
            "enterprise_value": data.get("enterprise_value"),
            "ebitda": data.get("ebitda"),
            "ev_ebitda": data.get("ev_ebitda"),
        }
