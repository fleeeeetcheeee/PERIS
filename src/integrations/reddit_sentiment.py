from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import BaseIntegration

FINANCE_SUBREDDITS = [
    "investing",
    "wallstreetbets",
    "stocks",
    "SecurityAnalysis",
    "finance",
    "privateequity",
]


class RedditSentimentIntegration(BaseIntegration):
    """PRAW-based Reddit sentiment scraper with VADER scoring."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or "https://oauth.reddit.com")
        self._analyzer = SentimentIntensityAnalyzer()
        self._reddit: praw.Reddit | None = None

    def _get_reddit(self) -> praw.Reddit:
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID", ""),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
                user_agent="PERIS:v0.1 (by /u/peris_bot)",
                read_only=True,
            )
        return self._reddit

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_sync(
        self,
        query: str,
        subreddit: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        reddit = self._get_reddit()
        target = reddit.subreddit(subreddit or "+".join(FINANCE_SUBREDDITS))
        posts = []
        for submission in target.search(query, limit=limit, sort="new", time_filter="month"):
            posts.append({
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext[:500],
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "url": submission.url,
                "subreddit": str(submission.subreddit),
                "created_utc": submission.created_utc,
            })
        return posts

    async def search_posts(
        self,
        query: str,
        subreddit: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Search Reddit posts relevant to a company or topic."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._search_sync, query, subreddit, limit
        )

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> float:
        """Return compound VADER score (-1.0 → 1.0) for a text string."""
        scores = self._analyzer.polarity_scores(text)
        return scores["compound"]

    async def analyze_sentiment(
        self, posts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Score each post and return aggregated sentiment signal.
        Returns:
            {
              "avg_compound": float,        # mean VADER compound score
              "positive_pct": float,        # % posts with compound > 0.05
              "negative_pct": float,
              "neutral_pct": float,
              "post_count": int,
              "scored_posts": list[dict]    # per-post scores
            }
        """
        if not posts:
            return {
                "avg_compound": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "post_count": 0,
                "scored_posts": [],
            }

        scored: list[dict[str, Any]] = []
        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}"
            compound = self._score_text(text)
            scored.append({**post, "compound_score": compound})

        compounds = [p["compound_score"] for p in scored]
        n = len(compounds)
        avg = sum(compounds) / n
        pos_pct = sum(1 for c in compounds if c > 0.05) / n * 100
        neg_pct = sum(1 for c in compounds if c < -0.05) / n * 100
        neu_pct = 100 - pos_pct - neg_pct

        return {
            "avg_compound": round(avg, 4),
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "neutral_pct": round(neu_pct, 1),
            "post_count": n,
            "scored_posts": scored,
        }

    # ------------------------------------------------------------------
    # Convenience: full pipeline for a company name
    # ------------------------------------------------------------------

    async def get_company_sentiment(
        self, company_name: str, limit: int = 25
    ) -> dict[str, Any]:
        """Search for a company name on Reddit and return sentiment signal."""
        posts = await self.search_posts(company_name, limit=limit)
        sentiment = await self.analyze_sentiment(posts)
        sentiment["query"] = company_name
        return sentiment
