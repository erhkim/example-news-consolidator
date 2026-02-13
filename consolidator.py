"""
Event consolidation module.

Problem: Multiple news sources report the same disruption. We need to merge
them into a single event with multiple source URLs.

Approach:
1. Embedding-based similarity clustering (fast, deterministic)
2. LLM confirmation + merge for borderline cases
3. Field-level merge logic (union locations, max severity, collect URLs)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"

# -- Similarity threshold tuning --
# Too low = separate events get merged (dangerous)
# Too high = same event stays split (annoying but safe)
SIMILARITY_THRESHOLD = 0.82


@dataclass
class ConsolidatedEvent:
    """A single disruption event, possibly sourced from multiple articles."""
    id: str
    title: str
    summary: str
    category: str
    severity: str
    transport_modes: list[str]
    affected_locations: list[dict]
    affected_trade_lanes: list[str]
    estimated_duration: str
    keywords: list[str]
    sources: list[dict]  # [{"url": ..., "name": ..., "published_at": ..., "relevance_score": ...}]
    first_seen: str
    last_updated: str
    article_count: int


class EventConsolidator:
    """Clusters and merges disruption results into deduplicated events."""

    def __init__(self, openai_api_key: str, use_llm_confirmation: bool = True):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.use_llm_confirmation = use_llm_confirmation

    async def consolidate(self, disruptions: list[dict]) -> list[ConsolidatedEvent]:
        """
        Main entry point. Takes raw disruption dicts from the analyzer,
        returns consolidated events.
        """
        if not disruptions:
            return []

        if len(disruptions) == 1:
            return [self._single_to_event(disruptions[0])]

        # Step 1: Build similarity matrix
        embeddings = self._compute_embeddings(disruptions)
        clusters = self._cluster_by_similarity(disruptions, embeddings)

        logger.info(
            f"Clustered {len(disruptions)} disruptions into {len(clusters)} groups"
        )

        # Step 2: Merge each cluster
        events = []
        for cluster in clusters:
            if len(cluster) == 1:
                events.append(self._single_to_event(cluster[0]))
            else:
                # Optional: LLM confirmation that these are truly the same event
                if self.use_llm_confirmation:
                    confirmed = await self._llm_confirm_cluster(cluster)
                    # If LLM says they're different events, split them
                    for sub_cluster in confirmed:
                        events.append(self._merge_cluster(sub_cluster))
                else:
                    events.append(self._merge_cluster(cluster))

        logger.info(f"Final consolidated events: {len(events)}")
        return events

    # -------------------------------------------------------------------------
    # Similarity / Clustering
    # -------------------------------------------------------------------------

    def _compute_embeddings(self, disruptions: list[dict]) -> np.ndarray:
        """
        Compute a lightweight fingerprint vector for each disruption.

        In production, use a real embedding model (e.g. voyage-3, text-embedding-3-small).
        This implementation uses a fast TF-IDF-like bag-of-words approach as a
        zero-dependency fallback.
        """
        # Build fingerprint from key fields
        texts = []
        for d in disruptions:
            fingerprint_parts = [
                d.get("category", ""),
                d.get("summary", ""),
                # Location fingerprint: country codes + city names
                " ".join(
                    f"{loc.get('country_code', '')} {loc.get('city', '')}"
                    for loc in d.get("affected_locations", [])
                ),
                # Transport modes
                " ".join(d.get("transport_modes", [])),
                # Keywords
                " ".join(d.get("keywords", [])),
            ]
            texts.append(" ".join(fingerprint_parts).lower())

        # Simple bag-of-words vectors (swap for real embeddings in prod)
        vocab = {}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        vectors = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            for word in text.split():
                vectors[i, vocab[word]] += 1
            # L2 normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        return vectors

    def _cluster_by_similarity(
        self, disruptions: list[dict], embeddings: np.ndarray
    ) -> list[list[dict]]:
        """
        Single-linkage clustering with similarity threshold.
        Simple and predictable. No need for DBSCAN overhead here.
        """
        n = len(disruptions)
        # Cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T

        # Also boost similarity if same category + overlapping locations
        for i in range(n):
            for j in range(i + 1, n):
                bonus = self._structural_similarity_bonus(
                    disruptions[i], disruptions[j]
                )
                sim_matrix[i, j] += bonus
                sim_matrix[j, i] += bonus

        # Union-Find clustering
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                    union(i, j)

        # Group by cluster
        clusters: dict[int, list[dict]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(disruptions[i])

        return list(clusters.values())

    def _structural_similarity_bonus(self, a: dict, b: dict) -> float:
        """
        Bonus similarity for matching structured fields.
        This helps catch cases where text similarity is moderate but
        the events clearly overlap (same category, same locations).
        """
        bonus = 0.0

        # Same category
        if a.get("category") == b.get("category"):
            bonus += 0.05

        # Overlapping country codes
        countries_a = {
            loc.get("country_code")
            for loc in a.get("affected_locations", [])
            if loc.get("country_code")
        }
        countries_b = {
            loc.get("country_code")
            for loc in b.get("affected_locations", [])
            if loc.get("country_code")
        }
        if countries_a & countries_b:
            bonus += 0.08

        # Overlapping IATA codes
        iata_a = {
            code
            for loc in a.get("affected_locations", [])
            for code in loc.get("iata_codes", [])
        }
        iata_b = {
            code
            for loc in b.get("affected_locations", [])
            for code in loc.get("iata_codes", [])
        }
        if iata_a & iata_b:
            bonus += 0.10

        return bonus

    # -------------------------------------------------------------------------
    # LLM Confirmation
    # -------------------------------------------------------------------------

    async def _llm_confirm_cluster(
        self, cluster: list[dict]
    ) -> list[list[dict]]:
        """
        Ask LLM to confirm whether clustered articles are about the same event.
        Returns sub-clusters (usually just one, but may split if LLM disagrees).
        """
        if len(cluster) <= 1:
            return [cluster]

        summaries = []
        for i, d in enumerate(cluster):
            summaries.append(
                f"[{i}] {d.get('title', 'No title')}: {d.get('summary', '')}"
            )

        prompt = f"""These articles were grouped as potentially describing the same freight disruption event.
Confirm whether they are about the SAME event or DIFFERENT events.

Articles:
{chr(10).join(summaries)}

Respond with ONLY valid JSON:
{{
  "same_event": true/false,
  "groups": [[0, 1, 2]] or [[0, 1], [2]] if different events,
  "reason": "<brief explanation>"
}}

Rules:
- Same event = same root cause, same general location, same timeframe
- Different framing or different sources covering the same incident = SAME event
- Related but distinct events (e.g. two different port closures) = DIFFERENT events"""

        try:
            response = await self.client.responses.create(
                model=MODEL,
                max_output_tokens=300,
                input=prompt,
            )
            import json
            raw = response.output_text
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(raw)

            groups = result.get("groups", [list(range(len(cluster)))])
            return [[cluster[i] for i in group] for group in groups]

        except Exception as e:
            logger.warning(f"LLM confirmation failed, keeping cluster as-is: {e}")
            return [cluster]

    # -------------------------------------------------------------------------
    # Merging
    # -------------------------------------------------------------------------

    def _merge_cluster(self, cluster: list[dict]) -> ConsolidatedEvent:
        """
        Merge multiple disruption dicts about the same event into one
        ConsolidatedEvent. Field-level merge strategy:

        - title: pick from highest relevance_score article
        - summary: pick longest (most detailed)
        - category: majority vote
        - severity: take the MAX (worst case)
        - transport_modes: UNION
        - affected_locations: UNION (deduplicated by country_code + city)
        - affected_trade_lanes: UNION
        - keywords: UNION
        - sources: COLLECT all URLs
        - estimated_duration: pick most specific
        """
        # Sort by relevance score descending
        ranked = sorted(
            cluster,
            key=lambda d: d.get("relevance_score", 0),
            reverse=True,
        )
        best = ranked[0]

        # Title: from highest scored article
        title = best.get("title", "Unknown Event")

        # Summary: longest one (most detailed)
        summary = max(
            (d.get("summary", "") for d in cluster),
            key=len,
        )

        # Category: majority vote
        from collections import Counter
        category_votes = Counter(d.get("category", "other") for d in cluster)
        category = category_votes.most_common(1)[0][0]

        # Severity: worst case
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        severity = max(
            (d.get("severity", "low") for d in cluster),
            key=lambda s: severity_order.get(s, 0),
        )

        # Transport modes: union
        transport_modes = list({
            mode
            for d in cluster
            for mode in d.get("transport_modes", [])
        })

        # Affected locations: union, deduped by (country_code, city)
        affected_locations = self._merge_locations(cluster)

        # Trade lanes: union
        trade_lanes = list({
            lane
            for d in cluster
            for lane in d.get("affected_trade_lanes", [])
        })

        # Keywords: union
        keywords = list({
            kw
            for d in cluster
            for kw in d.get("keywords", [])
        })

        # Duration: pick most specific (longest string as heuristic)
        estimated_duration = max(
            (d.get("estimated_duration", "unknown") for d in cluster),
            key=lambda s: len(s) if s != "unknown" else 0,
        )

        # Sources: collect all
        sources = []
        for d in cluster:
            sources.append({
                "url": d.get("source_url", ""),
                "name": d.get("source_name", ""),
                "published_at": d.get("published_at", ""),
                "relevance_score": d.get("relevance_score", 0),
            })

        # Timestamps
        published_dates = [
            d.get("published_at", "") for d in cluster if d.get("published_at")
        ]
        first_seen = min(published_dates) if published_dates else ""
        last_updated = max(published_dates) if published_dates else ""

        return ConsolidatedEvent(
            id=best.get("id", ""),
            title=title,
            summary=summary,
            category=category,
            severity=severity,
            transport_modes=transport_modes,
            affected_locations=affected_locations,
            affected_trade_lanes=trade_lanes,
            estimated_duration=estimated_duration,
            keywords=keywords,
            sources=sources,
            first_seen=first_seen,
            last_updated=last_updated,
            article_count=len(cluster),
        )

    def _merge_locations(self, cluster: list[dict]) -> list[dict]:
        """
        Merge affected_locations across articles. Deduplicate by
        (country_code, city), union IATA and UNLOCODE within each.
        """
        merged: dict[tuple, dict] = {}

        for d in cluster:
            for loc in d.get("affected_locations", []):
                key = (
                    loc.get("country_code", ""),
                    (loc.get("city") or "").lower(),
                )
                if key not in merged:
                    merged[key] = {
                        "country": loc.get("country"),
                        "country_code": loc.get("country_code"),
                        "region": loc.get("region"),
                        "city": loc.get("city"),
                        "iata_codes": set(),
                        "unlocode": set(),
                    }

                # Union codes
                for code in loc.get("iata_codes", []):
                    merged[key]["iata_codes"].add(code)
                for code in loc.get("unlocode", []):
                    merged[key]["unlocode"].add(code)

                # Fill in region if one article has it and another doesn't
                if loc.get("region") and not merged[key]["region"]:
                    merged[key]["region"] = loc["region"]

        # Convert sets back to lists
        result = []
        for loc in merged.values():
            loc["iata_codes"] = sorted(loc["iata_codes"])
            loc["unlocode"] = sorted(loc["unlocode"])
            result.append(loc)

        return result

    def _single_to_event(self, d: dict) -> ConsolidatedEvent:
        """Wrap a single disruption dict as a ConsolidatedEvent."""
        return ConsolidatedEvent(
            id=d.get("id", ""),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            category=d.get("category", "other"),
            severity=d.get("severity", "low"),
            transport_modes=d.get("transport_modes", []),
            affected_locations=d.get("affected_locations", []),
            affected_trade_lanes=d.get("affected_trade_lanes", []),
            estimated_duration=d.get("estimated_duration", "unknown"),
            keywords=d.get("keywords", []),
            sources=[{
                "url": d.get("source_url", ""),
                "name": d.get("source_name", ""),
                "published_at": d.get("published_at", ""),
                "relevance_score": d.get("relevance_score", 0),
            }],
            first_seen=d.get("published_at", ""),
            last_updated=d.get("fetched_at", ""),
            article_count=1,
        )
