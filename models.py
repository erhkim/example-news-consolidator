"""
Data models and configuration for the Freight Disruption Detection System.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


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
