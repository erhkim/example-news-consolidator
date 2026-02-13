# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Freight/logistics disruption detection system. An async Python pipeline that ingests news from multiple APIs, uses LLM-based analysis to identify supply chain disruptions, and clusters related events using embeddings.

## Running

```bash
# Requires Python 3.13
# Environment variables: OPENAI_API_KEY (required), NEWSAPI_KEY (optional)

# Single run
RUN_ONCE=1 python main.py

# Continuous hourly loop (default)
python main.py
```

Other env vars: `OUTPUT_DIR` (default `./output`), `RUN_INTERVAL_SECONDS` (default 3600), `MAX_CONCURRENT` (default 5).

No test suite or linting configuration exists currently.

## Architecture

**5-stage pipeline** orchestrated by `main.py`:

```
News Ingestion → Keyword Pre-Filter → LLM Relevance Scoring → LLM Classification+Extraction → Event Consolidation → JSON Output
```

### Core Modules

- **`main.py`** — Async pipeline orchestrator. Coordinates all stages, manages concurrency via semaphores, writes JSON output per run.
- **`ingestion.py`** (`NewsIngester`) — Fetches from NewsAPI (freight-related queries) and GDELT (themed event searches). Hash-based deduplication. Keyword pre-filter requires both a transport term AND a disruption term.
- **`analyzer.py`** (`DisruptionAnalyzer`) — 3-stage LLM pipeline using gpt-4o: (1) relevance scoring with 0.6 threshold, (2+3) classification and structured impact extraction via Pydantic `response.parse()`. Includes `CodeValidator` for IATA/UN-LOCODE validation against sample databases.
- **`consolidator.py`** (`EventConsolidator`) — Clusters related articles using `text-embedding-3-small` embeddings + structural similarity bonuses (same category, overlapping codes). Union-Find algorithm with 0.82 similarity threshold. Field-level merge strategy (e.g., severity=max, summary=longest, locations=union).
- **`models.py`** — Enums (`DisruptionCategory`, `SeverityLevel`, `TransportMode`), dataclasses (`AffectedLocation`, `DisruptionEvent`), and Pydantic models for structured OpenAI output (`RelevanceResult`, `DisruptionExtraction`, `LocationExtraction`, `ClusterConfirmation`).

### Key Design Patterns

- All OpenAI structured output uses Pydantic models with `response.parse()` (not raw JSON)
- Async throughout with `httpx.AsyncClient` and `openai.AsyncOpenAI`
- IATA/UN-LOCODE validators use hardcoded sample sets; production should load full OurAirports CSV and UN/LOCODE files
- GDELT works without an API key; NewsAPI requires `NEWSAPI_KEY`
