"""
News ingestion layer.

Fetches articles from multiple sources hourly, deduplicates,
and applies fast keyword pre-filtering before LLM classification.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from models import FREIGHT_KEYWORDS, NEWS_API_SOURCES

logger = logging.getLogger(__name__)


class NewsIngester:
    """Pulls news from configured sources and pre-filters for freight relevance."""

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        lookback_hours: int = 2,  # overlap to catch late-indexed articles
    ):
        self.newsapi_key = newsapi_key
        self.lookback_hours = lookback_hours
        self._seen_hashes: set[str] = set()  # simple in-memory dedup; swap for Redis in prod

    async def fetch_all(self) -> list[dict]:
        """Fetch from all configured sources, deduplicate, pre-filter."""
        raw_articles = []

        if self.newsapi_key:
            raw_articles.extend(await self._fetch_newsapi())

        raw_articles.extend(await self._fetch_gdelt())

        # Deduplicate
        unique = self._deduplicate(raw_articles)
        logger.info(f"Fetched {len(raw_articles)} raw, {len(unique)} after dedup")

        # Fast keyword pre-filter (cheap, before LLM)
        filtered = [a for a in unique if self._keyword_prefilter(a)]
        logger.info(f"{len(filtered)} articles passed keyword pre-filter")

        return filtered

    async def _fetch_newsapi(self) -> list[dict]:
        """Fetch from NewsAPI using freight-related queries."""
        config = NEWS_API_SOURCES["newsapi"]
        since = (datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)).isoformat()
        articles = []

        async with httpx.AsyncClient(timeout=30) as client:
            for query in config["freight_queries"]:
                try:
                    resp = await client.get(
                        config["base_url"],
                        params={
                            "q": query,
                            "from": since,
                            "sortBy": "publishedAt",
                            "language": "en",
                            "pageSize": 20,
                            "apiKey": self.newsapi_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for art in data.get("articles", []):
                        articles.append({
                            "title": art.get("title", ""),
                            "description": art.get("description", ""),
                            "content": art.get("content", ""),
                            "url": art.get("url", ""),
                            "source": art.get("source", {}).get("name", "unknown"),
                            "published_at": art.get("publishedAt", ""),
                        })
                except Exception as e:
                    logger.warning(f"NewsAPI query '{query}' failed: {e}")

        return articles

    async def _fetch_gdelt(self) -> list[dict]:
        """Fetch from GDELT (free, no API key needed)."""
        config = NEWS_API_SOURCES["gdelt"]
        articles = []

        async with httpx.AsyncClient(timeout=30) as client:
            for theme in config["themes"]:
                try:
                    resp = await client.get(
                        config["base_url"],
                        params={
                            "query": f"theme:{theme}",
                            "mode": "ArtList",
                            "maxrecords": 50,
                            "format": "json",
                            "timespan": f"{self.lookback_hours}h",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for art in data.get("articles", []):
                        articles.append({
                            "title": art.get("title", ""),
                            "description": art.get("seendate", ""),
                            "content": "",  # GDELT doesn't give full text
                            "url": art.get("url", ""),
                            "source": art.get("domain", "gdelt"),
                            "published_at": art.get("seendate", ""),
                        })
                except Exception as e:
                    logger.warning(f"GDELT theme '{theme}' failed: {e}")

        return articles

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """Hash-based dedup on title + URL."""
        unique = []
        for art in articles:
            h = hashlib.md5(
                f"{art['title']}|{art['url']}".encode()
            ).hexdigest()
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                unique.append(art)
        return unique

    def _keyword_prefilter(self, article: dict) -> bool:
        """
        Fast check: does the article mention at least one transport term
        AND one disruption term? This is a cheap gate before the LLM.
        """
        text = (
            f"{article.get('title', '')} {article.get('description', '')} "
            f"{article.get('content', '')}"
        ).lower()

        has_transport = any(
            kw in text for kw in FREIGHT_KEYWORDS["transport_terms"]
        )
        has_disruption = any(
            kw in text for kw in FREIGHT_KEYWORDS["disruption_terms"]
        )

        return has_transport and has_disruption
