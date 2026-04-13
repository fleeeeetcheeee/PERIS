"""Tests for data ingestion integrations.

Uses real HTTP calls where possible (EDGAR, feedparser) and mocks for
rate-limited or credentialed sources (FRED, Reddit, yfinance).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------

class TestSecEdgarIntegration:
    def test_parse_filing_hit(self):
        from src.integrations.sec_edgar import SecEdgarIntegration

        client = SecEdgarIntegration()
        hit = {
            "_id": "0001234567-24-000001:ex1.htm",
            "_source": {
                "entity_name": "Acme Corp",
                "form_type": "8-K",
                "file_date": "2024-01-15",
                "period_of_report": "2024-01-14",
            },
        }
        result = client._parse_filing_hit(hit)
        assert result["entity_name"] == "Acme Corp"
        assert result["form_type"] == "8-K"
        assert result["file_date"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_search_filings_live(self):
        """Light live call to EDGAR EFTS — no auth required."""
        from src.integrations.sec_edgar import SecEdgarIntegration

        client = SecEdgarIntegration()
        results = await client.search_filings("merger", form_type="8-K", limit=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for item in results:
            assert "file_date" in item

    @pytest.mark.asyncio
    async def test_get_company_filings_mock(self):
        from src.integrations.sec_edgar import SecEdgarIntegration

        mock_response = {
            "name": "Test Corp",
            "filings": {
                "recent": {
                    "form": ["8-K", "10-K"],
                    "filingDate": ["2024-03-01", "2024-02-01"],
                    "accessionNumber": ["0001-24-001", "0001-24-002"],
                    "primaryDocument": ["ex1.htm", "form10k.htm"],
                }
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)

            client = SecEdgarIntegration()
            filings = await client.get_company_filings("320193", limit=5)
            assert isinstance(filings, list)


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

class TestFredIntegration:
    def test_parse_value(self):
        from src.integrations.fred import FredIntegration

        client = FredIntegration()
        assert client._parse_value("28000.5") == 28000.5
        assert client._parse_value(".") is None
        assert client._parse_value("") is None

    @pytest.mark.asyncio
    async def test_get_series_observations_mock(self):
        from src.integrations.fred import FredIntegration

        mock_data = {
            "observations": [
                {"date": "2024-01-01", "value": "28000.0"},
                {"date": "2023-10-01", "value": "27500.0"},
                {"date": "2023-07-01", "value": "."},  # missing value
            ]
        }
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_data
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)

            client = FredIntegration(api_key="test")
            obs = await client.get_series_observations("GDP", limit=10)

            # Missing "." values should be filtered out
            assert len(obs) == 2
            assert obs[0]["value"] == 28000.0
            assert obs[1]["value"] == 27500.0


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

class TestYahooFinanceIntegration:
    def test_fetch_ticker_data_mock(self):
        from src.integrations.yahoo_finance import YahooFinanceIntegration
        import yfinance as yf

        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "currentPrice": 195.0,
            "marketCap": 3_000_000_000_000,
            "totalRevenue": 390_000_000_000,
            "ebitda": 130_000_000_000,
            "enterpriseValue": 3_100_000_000_000,
            "enterpriseToEbitda": 23.8,
        }

        with patch.object(yf.Ticker, "__init__", return_value=None):
            with patch.object(yf.Ticker, "info", new_callable=lambda: property(lambda self: mock_info)):
                client = YahooFinanceIntegration()
                result = client._fetch_ticker_data("AAPL")
                assert result["name"] == "Apple Inc."
                assert result["price"] == 195.0
                assert result["market_cap"] == 3_000_000_000_000

    @pytest.mark.asyncio
    async def test_get_quote_mock(self):
        from src.integrations.yahoo_finance import YahooFinanceIntegration

        client = YahooFinanceIntegration()
        expected = {"symbol": "MSFT", "name": "Microsoft", "price": 400.0}

        with patch.object(client, "_fetch_ticker_data", return_value=expected):
            result = await client.get_quote("MSFT")
            assert result["name"] == "Microsoft"


# ---------------------------------------------------------------------------
# RSS Feeds
# ---------------------------------------------------------------------------

class TestRSSFeedsIntegration:
    def test_classify_signal_8k(self):
        from src.integrations.rss_feeds import RSSFeedsIntegration

        client = RSSFeedsIntegration()
        entry = MagicMock()
        entry.title = "SEC Form 8-K filing"
        signal = client._classify_signal("https://sec.gov/8-k", entry)
        assert signal == "sec_8k"

    def test_classify_signal_ma(self):
        from src.integrations.rss_feeds import RSSFeedsIntegration

        client = RSSFeedsIntegration()
        entry = MagicMock()
        entry.title = "Company A acquires Company B"
        signal = client._classify_signal("https://feeds.reuters.com/business", entry)
        assert signal == "ma_news"

    def test_get_summary_strips_html(self):
        from src.integrations.rss_feeds import RSSFeedsIntegration

        client = RSSFeedsIntegration()
        entry = MagicMock()
        entry.summary = "<p>This is <b>important</b> news about <a href='#'>acquisitions</a>.</p>"
        result = client._get_summary(entry)
        assert "<" not in result
        assert "important" in result

    @pytest.mark.asyncio
    async def test_fetch_feed_mock(self):
        from src.integrations.rss_feeds import RSSFeedsIntegration
        import feedparser

        mock_entry = MagicMock()
        mock_entry.title = "Test M&A News"
        mock_entry.link = "https://example.com/news"
        mock_entry.summary = "Company acquired another company."
        mock_entry.tags = []
        mock_entry.published = "Mon, 01 Jan 2024 12:00:00 +0000"

        mock_parsed = MagicMock()
        mock_parsed.entries = [mock_entry]

        with patch("feedparser.parse", return_value=mock_parsed):
            client = RSSFeedsIntegration()
            items = await client.fetch_feed("https://example.com/feed")
            assert len(items) == 1
            assert items[0]["title"] == "Test M&A News"


# ---------------------------------------------------------------------------
# Reddit Sentiment
# ---------------------------------------------------------------------------

class TestRedditSentimentIntegration:
    def test_score_text_positive(self):
        from src.integrations.reddit_sentiment import RedditSentimentIntegration

        client = RedditSentimentIntegration.__new__(RedditSentimentIntegration)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        client._analyzer = SentimentIntensityAnalyzer()
        client._reddit = None

        score = client._score_text("This company is absolutely amazing and growing fast!")
        assert score > 0.05

    def test_score_text_negative(self):
        from src.integrations.reddit_sentiment import RedditSentimentIntegration

        client = RedditSentimentIntegration.__new__(RedditSentimentIntegration)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        client._analyzer = SentimentIntensityAnalyzer()
        client._reddit = None

        score = client._score_text("Terrible performance, massive losses, complete disaster.")
        assert score < -0.05

    @pytest.mark.asyncio
    async def test_analyze_sentiment_empty(self):
        from src.integrations.reddit_sentiment import RedditSentimentIntegration

        client = RedditSentimentIntegration.__new__(RedditSentimentIntegration)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        client._analyzer = SentimentIntensityAnalyzer()
        client._reddit = None

        result = await client.analyze_sentiment([])
        assert result["post_count"] == 0
        assert result["avg_compound"] == 0.0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_batch(self):
        from src.integrations.reddit_sentiment import RedditSentimentIntegration

        client = RedditSentimentIntegration.__new__(RedditSentimentIntegration)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        client._analyzer = SentimentIntensityAnalyzer()
        client._reddit = None

        posts = [
            {"title": "Great company, bullish!", "selftext": ""},
            {"title": "Terrible results, avoid.", "selftext": ""},
            {"title": "Neutral update.", "selftext": ""},
        ]
        result = await client.analyze_sentiment(posts)
        assert result["post_count"] == 3
        assert "positive_pct" in result
        assert "negative_pct" in result
        assert result["positive_pct"] + result["negative_pct"] + result["neutral_pct"] == pytest.approx(100.0, abs=1.0)
