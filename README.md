# Freight Disruption Detection System

## Architecture

```
News Sources (hourly)
    │
    ├── NewsAPI (paid, broader coverage)
    ├── GDELT (free, global events)
    └── [extensible: RSS feeds, Reuters API, etc.]
    │
    ▼
Keyword Pre-Filter (fast, cheap)
    │  Requires BOTH a transport term AND a disruption term
    │  Eliminates ~80% of irrelevant articles before LLM
    ▼
LLM Stage 1: Relevance Scoring
    │  Score 0.0-1.0, threshold at 0.6
    │  Could be replaced with fine-tuned classifier at scale
    ▼
LLM Stage 2+3: Classification + Entity Extraction (single call)
    │  Category, severity, transport modes
    │  Countries, regions, cities
    │  IATA airport codes, UN/LOCODE port codes
    │  Trade lanes, duration estimates
    ▼
Code Validation
    │  Cross-reference extracted IATA/UNLOCODE against known databases
    │  Strip invalid codes (LLMs hallucinate codes sometimes)
    ▼
Structured JSON Output
    └── Per-run files in ./output/
```

## Key Design Decisions

1. **Two-tier filtering**: Keyword pre-filter is essentially free and eliminates most noise before expensive LLM calls. At 1000 articles/hour, this saves significant API cost.

2. **Combined classification + extraction**: One LLM call instead of two. The model handles both tasks well in a single structured prompt, halving latency and cost.

3. **Code validation post-processing**: LLMs will occasionally hallucinate IATA/UNLOCODE values. The validator strips anything not in the known database. Load full databases from OurAirports CSV and UN/LOCODE files in production.

4. **GDELT as free baseline**: Works without API keys. NewsAPI adds breadth but costs money. System degrades gracefully with just GDELT.

5. **Concurrency-limited async**: Semaphore prevents blowing through API rate limits while still parallelizing analysis.

## Production Considerations

- **Persistent dedup**: Replace in-memory hash set with Redis/DB to survive restarts
- **Full IATA/UNLOCODE databases**: Load from CSV files (~40K IATA codes, ~100K UNLOCODEs)
- **Fine-tuned classifier for Stage 1**: At scale, a fine-tuned model (or even a logistic regression on embeddings) is cheaper than an LLM call per article
- **Article full-text fetching**: GDELT only gives URLs; use `newspaper3k` or `trafilatura` to scrape full text
- **Webhook/queue output**: Push results to Kafka, webhook, or database instead of flat files
- **Monitoring**: Track articles/hour, relevance distribution, category breakdown, LLM error rate
- **Caching**: Don't re-analyze the same article URL within 24h
- **Rate limiting**: NewsAPI free tier = 100 req/day; plan queries accordingly

## Setup

```bash
pip install httpx anthropic

export ANTHROPIC_API_KEY="sk-..."
export NEWSAPI_KEY="..."          # optional
export RUN_ONCE=1                  # single run, or omit for hourly loop

python main.py
```

## Output Schema

See `example_output.json` for full example. Each disruption contains:

| Field | Type | Description |
|-------|------|-------------|
| `category` | enum | weather, geopolitical, labor, infrastructure, regulatory, accident, pandemic_health, cyber, congestion, other |
| `severity` | enum | low, medium, high, critical |
| `transport_modes` | list | air, ocean, rail, road, intermodal |
| `affected_locations[].country_code` | str | ISO 3166-1 alpha-2 |
| `affected_locations[].iata_codes` | list | 3-letter airport codes |
| `affected_locations[].unlocode` | list | 5-char UN/LOCODE port codes |
| `affected_trade_lanes` | list | e.g. "Asia-US West Coast" |
| `estimated_duration` | str | human-readable estimate |

## Extending

**Add a news source**: Implement a `_fetch_<source>` method in `ingestion.py` returning the standard article dict format.

**Add a disruption category**: Update `DisruptionCategory` enum in `models.py` and the classification prompt in `analyzer.py`.

**Change LLM provider**: Swap the Anthropic client in `analyzer.py` for OpenAI, a local model, etc. The prompt structure stays the same.
