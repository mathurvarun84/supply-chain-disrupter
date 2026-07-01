# CLAUDE.md — Agent context for this repo

## Project

Supply Chain Disruption Predictor (Capstone P8). LangGraph pipeline: L1 ingestion → L2 news → L3 weather → L4 risk classifier → L5/L6 optional → L7 mitigation.

## Key layout

| Layer | Path | Role |
|-------|------|------|
| L1 | `src/agents/data_ingestion/live_ingest.py` | GDELT/RSS → `news_signals`; Open-Meteo → `weather_signals` |
| L2 | `src/agents/news_agent/agent.py` | SQLite-first news analysis + LLM |
| L3 | `src/agents/weather_agent/agent.py` | SQLite-first weather risk + LLM |
| L3 client | `src/agents/weather_agent/client.py` | Open-Meteo HTTP (used by L1 and L3 live fallback only) |
| Graph | `src/agents/langgraph_engine.py` | Wires L2/L3 from folder `agent.py` modules |
| State | `src/agents/state.py` | `GlobalState`, Pydantic LLM output models |
| DB | `src/utils/db_utils.py` | `fetch_recent_news`, `fetch_latest_weather_signal`, ingestion schema |
| RAG | `src/rag/` | ChromaDB; `build_news_signals()` is L2 tertiary fallback |

**Removed:** flat files `src/agents/news_agent.py`, `src/agents/weather_agent.py`, and `news_agent/rag.py` (shim). Single source of truth is each package’s `agent.py`.

## L2 News agent (SQLite-first)

1. `fetch_recent_news(region, limit=20)` — no live news API calls
2. `semiconductor_signals` query for order year
3. Three `build_rag_context()` queries
4. LLM (`MODEL_FAST` / gpt-4.1-mini) → `NewsAnalysisLLMOutput`
5. Signals: primary + regional (0.75×)
6. Fallback: `FALLBACK_PARAMS` (+0.05 if >5 news rows) → `build_news_signals()` for unknown types

Exports from package: `news_event_analysis_agent`, `NEWS_SYSTEM_PROMPT`, `build_news_signals`.

## L3 Weather agent (SQLite-first)

1. Coords from record or port config; `_nearest_hub()` (12 hubs)
2. `fetch_latest_weather_signal(hub)` — primary; no Open-Meteo unless no row
3. Live fallback: `client.fetch_open_meteo` + `compute_weather_severity`
4. RAG if severity ≥ 0.40
5. LLM → `geo_risk_component` overrides numeric severity
6. Fallback: numeric severity unchanged

## Tests

```bash
python -m pytest tests/test_news_weather_agents_v2.py -v
python -m pytest tests/test_llm_agents.py -v
```

Patch targets for mocks: `src.agents.news_agent.agent.*` and `src.agents.weather_agent.agent.*` (not package root).

## Docs

Full architecture: `docs/ARCHITECTURE.md`
