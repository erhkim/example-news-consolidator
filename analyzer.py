"""
LLM-based analysis pipeline.

Stage 1: Relevance scoring (is this actually about freight disruption?)
Stage 2: Classification (what kind of disruption?)
Stage 3: Impact extraction (where, what transport codes?)

Uses structured output via JSON mode to get reliable extraction.
"""

import json
import logging
from typing import Optional

from anthropic import AsyncAnthropic

from models import (
    AffectedLocation,
    DisruptionCategory,
    DisruptionEvent,
    SeverityLevel,
    TransportMode,
)

logger = logging.getLogger(__name__)

# You can swap this for OpenAI, local models, etc.
# Using Claude because structured extraction is a strength.
MODEL = "claude-sonnet-4-5-20250929"


class DisruptionAnalyzer:
    """Three-stage LLM pipeline for disruption detection and extraction."""

    def __init__(self, anthropic_api_key: str):
        self.client = AsyncAnthropic(api_key=anthropic_api_key)

    async def analyze_article(self, article: dict) -> Optional[dict]:
        """
        Full pipeline: relevance -> classification -> impact extraction.
        Returns structured dict or None if not relevant.
        """
        text = self._build_article_text(article)

        # Stage 1: Relevance check (cheap, fast — use haiku or a classifier)
        relevance = await self._check_relevance(text)
        if relevance["score"] < 0.6:
            logger.debug(f"Filtered out (score={relevance['score']}): {article.get('title')}")
            return None

        # Stage 2+3: Classification and extraction in one call
        # Combining saves latency and cost vs two separate calls
        result = await self._classify_and_extract(text)
        if not result:
            return None

        result["relevance_score"] = relevance["score"]
        result["source_url"] = article.get("url", "")
        result["source_name"] = article.get("source", "")
        result["title"] = article.get("title", "")
        result["published_at"] = article.get("published_at", "")

        return result

    async def _check_relevance(self, text: str) -> dict:
        """
        Stage 1: Quick relevance scoring.
        Could also be a fine-tuned classifier for cost savings at scale.
        """
        prompt = f"""Score this article's relevance to freight/logistics disruptions from 0.0 to 1.0.

Relevant = the article describes an event that could materially impact freight movement,
shipping schedules, port operations, air cargo, trucking, rail freight, or supply chains.

NOT relevant = general business news, stock market, tech news, politics without
direct transport/trade impact, local crime, sports, entertainment.

Article:
{text[:3000]}

Respond with ONLY valid JSON:
{{"score": <float>, "reason": "<one sentence>"}}"""

        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            return {"score": 0.0, "reason": "error"}

    async def _classify_and_extract(self, text: str) -> Optional[dict]:
        """
        Stage 2+3 combined: Classify disruption type and extract impact details.
        Single LLM call for efficiency.
        """
        prompt = f"""You are a freight logistics disruption analyst. Analyze this article and extract structured data.

Article:
{text[:4000]}

Respond with ONLY valid JSON matching this schema exactly:

{{
  "summary": "<2-3 sentence summary of the disruption>",
  "category": "<one of: weather, geopolitical, labor, infrastructure, regulatory, accident, pandemic_health, cyber, congestion, other>",
  "severity": "<one of: low, medium, high, critical>",
  "transport_modes": ["<affected modes: air, ocean, rail, road, intermodal>"],
  "affected_locations": [
    {{
      "country": "<full country name>",
      "country_code": "<ISO 3166-1 alpha-2, e.g. US, CN, DE>",
      "region": "<state/province/region or null>",
      "city": "<city name or null>",
      "iata_codes": ["<airport IATA codes if air transport affected, e.g. LAX, PVG>"],
      "unlocode": ["<UN/LOCODE for ports if ocean transport affected, e.g. USLAX, CNSHA>"]
    }}
  ],
  "affected_trade_lanes": ["<e.g. Asia-US West Coast, Europe-Middle East>"],
  "estimated_duration": "<e.g. 2-5 days, ongoing, unknown>",
  "keywords": ["<relevant keywords for indexing>"]
}}

Rules:
- Only include locations explicitly mentioned or strongly implied by the article.
- Use correct IATA codes (3-letter airport codes) and UN/LOCODEs (5-char port codes).
- If an article mentions a port city, include both the city and its port UNLOCODE.
- If uncertain about a code, omit it rather than guess.
- severity: critical = full stoppage/major route blocked, high = significant delays,
  medium = partial impact, low = potential/minor impact.
"""

        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text

            # Handle potential markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}\nRaw: {raw[:500]}")
            return None
        except Exception as e:
            logger.error(f"Classification/extraction failed: {e}")
            return None

    @staticmethod
    def _build_article_text(article: dict) -> str:
        """Combine article fields into a single text block for the LLM."""
        parts = []
        if article.get("title"):
            parts.append(f"Title: {article['title']}")
        if article.get("description"):
            parts.append(f"Description: {article['description']}")
        if article.get("content"):
            parts.append(f"Content: {article['content']}")
        return "\n".join(parts)


class CodeValidator:
    """
    Post-processing: validate IATA and UNLOCODE against known databases.
    In production, load from a real dataset (OurAirports, UN/LOCODE CSV).
    """

    def __init__(
        self,
        iata_db_path: Optional[str] = None,
        unlocode_db_path: Optional[str] = None,
    ):
        # In production, load these from CSV/DB
        # For now, a small sample for illustration
        self._known_iata = {
            "LAX", "JFK", "ORD", "ATL", "DFW", "SFO", "MIA", "SEA",
            "PVG", "SHA", "PEK", "HKG", "NRT", "ICN", "SIN", "BKK",
            "LHR", "FRA", "AMS", "CDG", "DXB", "DOH", "IST",
            "GRU", "MEX", "YYZ", "SYD", "MEM", "CVG", "SDF", "ANC",
            # ... load full list in production
        }
        self._known_unlocode = {
            "USLAX", "USNYC", "USSAV", "USHOU", "USLGB", "USOAK",
            "CNSHA", "CNNGB", "CNYTN", "CNSZX", "CNQIN",
            "SGSIN", "KRPUS", "JPYOK", "JPTYO", "TWKHH",
            "NLRTM", "DEHAM", "BEANR", "GBFXT", "GBSOU",
            "AEJEA", "AEAUH", "EGPSD",
            # ... load full list in production
        }

    def validate_and_clean(self, result: dict) -> dict:
        """Remove invalid codes from LLM output."""
        for loc in result.get("affected_locations", []):
            loc["iata_codes"] = [
                c for c in loc.get("iata_codes", [])
                if c.upper() in self._known_iata
            ]
            loc["unlocode"] = [
                c for c in loc.get("unlocode", [])
                if c.upper() in self._known_unlocode
            ]
        return result
