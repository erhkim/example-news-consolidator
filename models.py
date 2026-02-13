"""
Data models and configuration for the Freight Disruption Detection System.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field


class DisruptionCategory(str, Enum):
    WEATHER = "weather"
    GEOPOLITICAL = "geopolitical"
    LABOR = "labor"
    INFRASTRUCTURE = "infrastructure"
    REGULATORY = "regulatory"
    ACCIDENT = "accident"
    PANDEMIC_HEALTH = "pandemic_health"
    CYBER = "cyber"
    CONGESTION = "congestion"
    OTHER = "other"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransportMode(str, Enum):
    AIR = "air"
    OCEAN = "ocean"
    RAIL = "rail"
    ROAD = "road"
    INTERMODAL = "intermodal"


@dataclass
class AffectedLocation:
    country: Optional[str] = None
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2
    region: Optional[str] = None
    city: Optional[str] = None
    iata_codes: list[str] = field(default_factory=list)  # airports
    unlocode: list[str] = field(default_factory=list)  # UN/LOCODE for ports
    coordinates: Optional[dict] = None  # {"lat": float, "lon": float}


@dataclass
class DisruptionEvent:
    id: str
    title: str
    summary: str
    source_url: str
    source_name: str
    published_at: datetime
    fetched_at: datetime
    relevance_score: float  # 0.0 - 1.0
    category: DisruptionCategory
    severity: SeverityLevel
    transport_modes: list[TransportMode]
    affected_locations: list[AffectedLocation]
    estimated_duration: Optional[str] = None  # e.g. "2-5 days"
    keywords: list[str] = field(default_factory=list)
    raw_text: str = ""


# -- Pydantic models for OpenAI Structured Output --

class RelevanceResult(BaseModel):
    """Response schema for Stage 1: relevance scoring."""
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reason: str = Field(description="One-sentence explanation of the score")


class LocationExtraction(BaseModel):
    """A single affected location extracted from an article."""
    country: str = Field(description="Full country name")
    country_code: str = Field(description="ISO 3166-1 alpha-2 code, e.g. US, CN, DE")
    region: Optional[str] = Field(default=None, description="State/province/region, or null if unknown")
    city: Optional[str] = Field(default=None, description="City name, or null if unknown")
    iata_codes: list[str] = Field(default_factory=list, description="Airport IATA codes if air transport affected, e.g. LAX, PVG")
    unlocode: list[str] = Field(default_factory=list, description="UN/LOCODE for ports if ocean transport affected, e.g. USLAX, CNSHA")


class DisruptionExtraction(BaseModel):
    """Response schema for Stage 2+3: classification and impact extraction."""
    summary: str = Field(description="2-3 sentence summary of the disruption")
    category: DisruptionCategory = Field(description="Type of disruption")
    severity: SeverityLevel = Field(description="Severity level: critical = full stoppage/major route blocked, high = significant delays, medium = partial impact, low = potential/minor impact")
    transport_modes: list[TransportMode] = Field(description="Affected transport modes")
    affected_locations: list[LocationExtraction] = Field(description="Locations affected by the disruption")
    affected_trade_lanes: list[str] = Field(default_factory=list, description="Affected trade lanes, e.g. Asia-US West Coast, Europe-Middle East")
    estimated_duration: str = Field(description="Duration estimate, e.g. 2-5 days, ongoing, unknown")
    keywords: list[str] = Field(default_factory=list, description="Relevant keywords for indexing")


class ClusterConfirmation(BaseModel):
    """Response schema for consolidator LLM cluster confirmation."""
    same_event: bool = Field(description="Whether all articles describe the same event")
    groups: list[list[int]] = Field(description="Article index groups, e.g. [[0, 1, 2]] if same event, [[0, 1], [2]] if different")
    reason: str = Field(description="Brief explanation of the grouping decision")


# -- Configuration --

NEWS_API_SOURCES = {
    "newsapi": {
        "base_url": "https://newsapi.org/v2/everything",
        "freight_queries": [
            "freight disruption",
            "port congestion",
            "shipping delay",
            "supply chain disruption",
            "airport closure",
            "rail strike logistics",
            "trucking shortage",
            "canal blockage",
            "trade embargo",
            "customs delay",
            "warehouse fire logistics",
            "hurricane shipping",
            "typhoon port",
            "flood transportation",
        ],
    },
    "gdelt": {
        "base_url": "https://api.gdeltproject.org/api/v2/doc/doc",
        "themes": [
            "TRANSPORT",
            "SUPPLY_CHAIN",
            "NATURAL_DISASTER",
            "TRADE",
        ],
    },
}

# Keywords for fast pre-filtering before LLM classification
FREIGHT_KEYWORDS = {
    "transport_terms": [
        "freight", "cargo", "shipping", "logistics", "supply chain",
        "port", "terminal", "warehouse", "container", "vessel",
        "trucking", "rail", "airfreight", "air cargo", "drayage",
        "intermodal", "last mile", "3pl", "forwarder", "customs",
        "import", "export", "tariff", "embargo", "trade route",
    ],
    "disruption_terms": [
        "delay", "disruption", "closure", "strike", "congestion",
        "blockage", "shutdown", "suspension", "diversion", "backlog",
        "shortage", "embargo", "sanction", "accident", "collision",
        "grounding", "derailment", "flood", "hurricane", "typhoon",
        "earthquake", "wildfire", "storm", "ice", "fog", "volcano",
    ],
}
