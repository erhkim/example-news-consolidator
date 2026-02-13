"""
LLM-based analysis pipeline.

Stage 1: Relevance scoring (is this actually about freight disruption?)
Stage 2: Classification (what kind of disruption?)
Stage 3: Impact extraction (where, what transport codes?)

Uses Pydantic structured output for reliable, typed extraction.
"""

import logging
from typing import Optional

from openai import AsyncOpenAI

from models import DisruptionExtraction, RelevanceResult

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"


class DisruptionAnalyzer:
    """Three-stage LLM pipeline for disruption detection and extraction."""

    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)

    async def analyze_article(self, article: dict) -> Optional[dict]:
        """
        Full pipeline: relevance -> classification -> impact extraction.
        Returns structured dict or None if not relevant.
        """
        text = self._build_article_text(article)

        # Stage 1: Relevance check (cheap, fast — use haiku or a classifier)
        relevance = await self._check_relevance(text)
        if relevance.score < 0.6:
            logger.debug(f"Filtered out (score={relevance.score}): {article.get('title')}")
            return None

        # Stage 2+3: Classification and extraction in one call
        # Combining saves latency and cost vs two separate calls
        extraction = await self._classify_and_extract(text)
        if not extraction:
            return None

        result = extraction.model_dump()
        result["relevance_score"] = relevance.score
        result["source_url"] = article.get("url", "")
        result["source_name"] = article.get("source", "")
        result["title"] = article.get("title", "")
        result["published_at"] = article.get("published_at", "")

        return result

    async def _check_relevance(self, text: str) -> RelevanceResult:
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
{text[:3000]}"""

        try:
            response = await self.client.responses.parse(
                model=MODEL,
                max_output_tokens=150,
                input=prompt,
                text_format=RelevanceResult,
            )
            return response.output_parsed
        except Exception as e:
            logger.error(f"Relevance check failed: {e}")
            return RelevanceResult(score=0.0, reason="error")

    async def _classify_and_extract(self, text: str) -> Optional[DisruptionExtraction]:
        """
        Stage 2+3 combined: Classify disruption type and extract impact details.
        Single LLM call for efficiency.
        """
        prompt = f"""You are a freight logistics disruption analyst. Analyze this article and extract structured data.

Article:
{text[:4000]}

Rules:
- Only include locations explicitly mentioned or strongly implied by the article.
- Use correct IATA codes (3-letter airport codes) and UN/LOCODEs (5-char port codes).
- If an article mentions a port city, include both the city and its port UNLOCODE.
- If uncertain about a code, omit it rather than guess."""

        try:
            response = await self.client.responses.parse(
                model=MODEL,
                max_output_tokens=1500,
                input=prompt,
                text_format=DisruptionExtraction,
            )
            return response.output_parsed
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
