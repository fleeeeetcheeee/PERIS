from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from .base import BaseIntegration

# Default feeds tracked by PERIS
DEFAULT_FEEDS = {
    "reuters_ma": "https://feeds.reuters.com/reuters/mergersNews",
    "sec_8k": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
}


class RSSFeedsIntegration(BaseIntegration):
    """Async RSS/Atom feed ingestion using feedparser."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or "")

    def parse(self, response: httpx.Response) -> Any:
        return feedparser.parse(response.text)

    # ------------------------------------------------------------------
    # Core fetch methods
    # ------------------------------------------------------------------

    async def fetch_feed(self, feed_url: str) -> list[dict[str, Any]]:
        """Fetch and parse a single RSS/Atom feed URL."""
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, feed_url)
        return [self._entry_to_dict(entry, feed_url) for entry in parsed.entries]

    async def fetch_all(self, feed_urls: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple feeds concurrently and return a flat list of items."""
        tasks = [self.fetch_feed(url) for url in feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            items.extend(result)
        return items

    async def fetch_default_feeds(self) -> list[dict[str, Any]]:
        """Fetch all default PERIS feeds (Reuters M&A + SEC 8-K alerts)."""
        return await self.fetch_all(list(DEFAULT_FEEDS.values()))

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _entry_to_dict(self, entry: Any, source_url: str) -> dict[str, Any]:
        published = self._parse_published(entry)
        return {
            "title": getattr(entry, "title", ""),
            "link": getattr(entry, "link", ""),
            "summary": self._get_summary(entry),
            "published": published,
            "source_url": source_url,
            "signal_type": self._classify_signal(source_url, entry),
            "tags": [t.get("term", "") for t in getattr(entry, "tags", [])],
        }

    def _get_summary(self, entry: Any) -> str:
        summary = getattr(entry, "summary", "") or ""
        # Strip minimal HTML
        import re
        return re.sub(r"<[^>]+>", " ", summary).strip()[:2000]

    def _parse_published(self, entry: Any) -> str:
        for attr in ("published", "updated", "created"):
            raw = getattr(entry, attr, None)
            if raw:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(raw)
                    return dt.astimezone(timezone.utc).isoformat()
                except Exception:
                    return raw
        return datetime.now(timezone.utc).isoformat()

    def _classify_signal(self, source_url: str, entry: Any) -> str:
        url_lower = source_url.lower()
        title_lower = getattr(entry, "title", "").lower()
        if "8-k" in url_lower or "8-k" in title_lower:
            return "sec_8k"
        if "merger" in url_lower or any(
            kw in title_lower for kw in ("acqui", "merger", "deal", "takeover", "buyout")
        ):
            return "ma_news"
        return "news"

    # ------------------------------------------------------------------
    # Signal formatting (for signals table)
    # ------------------------------------------------------------------

    def items_to_signals(
        self,
        items: list[dict[str, Any]],
        company_id: int,
    ) -> list[dict[str, Any]]:
        """Convert feed items to signal dicts ready for create_signal()."""
        signals = []
        for item in items:
            signals.append({
                "company_id": company_id,
                "signal_type": item["signal_type"],
                "summary": f"{item['title']} — {item['summary'][:300]}",
                "raw_data": item,
                "confidence": 0.6,
            })
        return signals
