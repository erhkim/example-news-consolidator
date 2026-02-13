"""
Main orchestrator for the Freight Disruption Detection System.

Runs hourly:
1. Ingest news from multiple sources
2. Pre-filter with keywords
3. Score relevance with LLM
4. Classify + extract structured impact data
5. Validate codes
6. Output / store results

Run: python main.py
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dataclasses import asdict

from ingestion import NewsIngester
from analyzer import DisruptionAnalyzer, CodeValidator
from consolidator import EventConsolidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("disruption_pipeline")

# -- Config from env --
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # optional, GDELT works without keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
RUN_INTERVAL_SECONDS = int(os.getenv("RUN_INTERVAL_SECONDS", "3600"))  # 1 hour
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT", "5"))


async def run_pipeline() -> list[dict]:
    """Single pipeline execution."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== Pipeline run {run_id} starting ===")

    # 1. Ingest
    ingester = NewsIngester(newsapi_key=NEWSAPI_KEY)
    articles = await ingester.fetch_all()
    logger.info(f"Pre-filtered articles to analyze: {len(articles)}")

    if not articles:
        logger.info("No relevant articles found this cycle.")
        return []

    # 2. Analyze with LLM (with concurrency limit)
    analyzer = DisruptionAnalyzer(openai_api_key=OPENAI_API_KEY)
    validator = CodeValidator()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

    async def _analyze_one(article: dict) -> dict | None:
        async with semaphore:
            result = await analyzer.analyze_article(article)
            if result:
                result = validator.validate_and_clean(result)
                result["id"] = str(uuid.uuid4())
                result["fetched_at"] = datetime.now(timezone.utc).isoformat()
            return result

    tasks = [_analyze_one(a) for a in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None and errors
    disruptions = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Analysis task failed: {r}")
        elif r is not None:
            disruptions.append(r)

    logger.info(f"Detected {len(disruptions)} disruptions from {len(articles)} articles")

    # 3. Consolidate duplicate events
    consolidator = EventConsolidator(openai_api_key=OPENAI_API_KEY)
    events = await consolidator.consolidate(disruptions)
    logger.info(
        f"Consolidated {len(disruptions)} disruptions into {len(events)} unique events"
    )

    # 4. Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"disruptions_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_at": datetime.now(timezone.utc).isoformat(),
                "total_articles_fetched": len(articles),
                "raw_disruptions_detected": len(disruptions),
                "consolidated_events": len(events),
                "events": [asdict(e) for e in events],
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Output saved to {output_file}")

    return events


async def main():
    """Run pipeline on a loop, or once if RUN_ONCE is set."""
    if os.getenv("RUN_ONCE"):
        await run_pipeline()
        return

    while True:
        try:
            await run_pipeline()
        except Exception as e:
            logger.exception(f"Pipeline run failed: {e}")

        logger.info(f"Sleeping {RUN_INTERVAL_SECONDS}s until next run...")
        await asyncio.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
