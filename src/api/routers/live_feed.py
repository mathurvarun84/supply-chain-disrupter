"""
Live Feed (Screen 1) endpoints — L1 only.

These routes are the ONLY endpoints in the whole app permitted to trigger
outbound network calls (via DataIngestionAgent). Every other screen
(L2-L7 — Risk Classification, Forecast & Simulation, Mitigation Plan,
Observability, RAG/RAGAS) is a pure SQLite/ChromaDB reader by design — see
ARCHITECTURE.md "L1 is the sole external I/O boundary". Do not add
LangGraph or LLM calls to this file; if a future screen needs them, that
work belongs in a different router (Days 3-9), not here.

Scope cuts (evaluator-facing, see also the plan doc):
  - The 7-agent LangGraph pipeline is never invoked here — only
    DataIngestionAgent.run_batch() runs, proving the L1 I/O boundary in
    isolation.
  - /logs and /gantt are hybrid: the L1 entry is real (from
    ingestion_run_log), the L2-L7 entries are the unmodified Day-1 fixture
    data from src/api/fixtures.py (LOG_LINES / GANTT) and become real on
    Day 9. Each entry is tagged source="real"|"stub" so the frontend can
    render the scope cut honestly instead of hiding it.
  - No hourly auto-scheduler here — manual "Refresh Live Data" (POST
    /refresh) plus passive 15s polling of already-ingested rows.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks

from src.agents.data_ingestion_agent import DataIngestionAgent, _INGESTION_LOCK
from src.api.fixtures import GANTT, LOG_LINES
from src.api.live_feed_schemas import (
    GanttBarResponse,
    IngestRunLogRow,
    IngestStatusResponse,
    LiveFeedGanttResponse,
    LiveFeedLogsResponse,
    LiveFeedNewsResponse,
    LiveFeedWeatherResponse,
    LogLineResponse,
    NewsHeadline,
    RefreshTriggeredResponse,
    WeatherHub,
)
from src.utils.db_utils import execute_query, fetch_live_news, fetch_live_weather, fetch_run_gantt, fetch_run_logs

router = APIRouter()

# L2-L7 entries carried over byte-for-byte from the Day-1 fixture — never
# edited here. Index 0 in each fixture list is the L1 entry, which this
# router replaces with real data; indices 1: are L2-L7 and stay as fixtures
# until Day 9 wires in the real LangGraph pipeline.
_STUB_LOG_LINES_L2_L7 = LOG_LINES[1:]
_STUB_GANTT_BARS_L2_L7 = GANTT[1:]


def _score_tier(score: Optional[float]) -> str:
    """Bucket a 0-1 relevance score into low/medium/high for the news score
    chip color. Server-computed so the threshold logic lives in one place.
    No existing project-wide relevance/severity constant was found in the
    repo (searched near risk_classifier_agent.py / guardrails.py) — these
    thresholds are new, screen-local defaults pending a project-wide
    constant; confirm with Varun before reusing elsewhere."""
    if score is None:
        return "low"
    if score > 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


@router.get("/news", response_model=LiveFeedNewsResponse)
def get_live_feed_news():
    """Return the most recent batch of ingested news headlines for Screen 1's
    News panel. Reads live_news_ingest via fetch_live_news() — never calls an
    external API or LangGraph/L2-L7."""
    payload = fetch_live_news(limit=50)
    if not payload["items"]:
        return LiveFeedNewsResponse(run_id=None, count=0, fetched_at=None, items=[])

    items = [
        NewsHeadline(
            hub_city=row["hub_city"],
            hub_country=row["hub_country"],
            supplier_country=row["supplier_country"],
            headline=row["headline"],
            published_at=row["published_at"],
            relevance_score=row["relevance_score"],
            query_term=row["query_term"],
            source_feed=row["source_feed"],
            score_tier=_score_tier(row["relevance_score"]),
        )
        for row in payload["items"]
    ]
    return LiveFeedNewsResponse(
        run_id=payload["run_id"],
        count=payload["count"],
        fetched_at=payload["fetched_at"],
        items=items,
    )


@router.get("/weather", response_model=LiveFeedWeatherResponse)
def get_live_feed_weather():
    """Return live severity for all 6 fab-hub cities via fetch_live_weather().

    is_trigger_hub is server-recomputed from raw_severity_score >= 7.0."""
    payload = fetch_live_weather()
    hubs = [
        WeatherHub(
            hub_city=hub["hub_city"],
            wind_speed_kmh=hub.get("wind_speed_kmh"),
            precipitation_mm=hub.get("precipitation_mm"),
            weather_code=hub.get("weather_code"),
            temperature_c=hub.get("temperature_c"),
            raw_severity_score=hub.get("raw_severity_score"),
            is_trigger_hub=bool(hub.get("is_trigger_hub")),
            fetched_at_utc=hub.get("fetched_at_utc"),
        )
        for hub in payload["hubs"]
    ]
    return LiveFeedWeatherResponse(
        run_id=payload["run_id"],
        fetched_at=payload["fetched_at"],
        hubs=hubs,
    )


@router.get("/ingest-status", response_model=IngestStatusResponse)
def get_ingest_status():
    """Return whether an ingestion batch is currently running plus the most
    recent `ingestion_run_log` rows. `is_running` reads DataIngestionAgent's
    own `_INGESTION_LOCK` directly rather than maintaining a second,
    independently-tracked flag (avoids two sources of truth). Used by
    RefreshControl to flip the "Refresh Live Data" button back on and to
    drive the frontend's fast-poll (2s) while a refresh is in flight."""
    rows = execute_query(
        """
        SELECT run_ts_utc, source, status, rows_inserted, rows_skipped,
               duration_ms, error_detail
        FROM ingestion_run_log
        ORDER BY run_ts_utc DESC LIMIT 20
        """
    )
    recent = [IngestRunLogRow(**dict(row)) for row in rows]
    return IngestStatusResponse(
        is_running=_INGESTION_LOCK.locked(),
        last_run=recent[0] if recent else None,
        recent_runs=recent,
    )


def _run_ingestion_batch() -> None:
    """Background-task body: runs one full DataIngestionAgent connector sweep
    (Open-Meteo, GDELT, FRED, Google News/Reuters RSS, CISA/BIS, YFinance).
    This is the ONLY function in this router that performs outbound network
    I/O — kept isolated here so the L1-boundary claim in ARCHITECTURE.md is
    easy to audit by grepping this file for network calls. Runs off the
    request thread via FastAPI's BackgroundTasks because a full connector
    sweep can take up to ~60s; never call this synchronously in a route."""
    agent = DataIngestionAgent()
    agent.run_batch()  # writes to live_news_ingest / live_weather_ingest /
    # ingestion_run_log; also self-guards via _INGESTION_LOCK.


@router.post("/refresh", response_model=RefreshTriggeredResponse)
def refresh_live_feed(background_tasks: BackgroundTasks):
    """Manually trigger one DataIngestionAgent batch (L1 only — never
    LangGraph/L2-L7). Fires-and-returns immediately; the caller polls
    `/api/live-feed/ingest-status` and then the news/weather GETs to see
    fresh data. Returns `skipped_already_running` instead of queuing a
    second overlapping batch if one is already in flight — checked against
    DataIngestionAgent's own lock, not a duplicate flag."""
    if _INGESTION_LOCK.locked():
        return RefreshTriggeredResponse(
            status="skipped_already_running",
            triggered_at=datetime.now(timezone.utc).isoformat(),
        )
    background_tasks.add_task(_run_ingestion_batch)
    return RefreshTriggeredResponse(
        status="started",
        triggered_at=datetime.now(timezone.utc).isoformat(),
    )


def _build_real_l1_log_line() -> LogLineResponse:
    """Build the one real Agent Log line from the latest ingestion_run_log
    batch. Only this function's output is allowed to replace the L1 entry —
    the L2-L7 fixture entries are appended unmodified by the caller."""
    latest = execute_query(
        "SELECT run_id FROM ingestion_run_log ORDER BY run_ts_utc DESC LIMIT 1"
    )
    if not latest:
        return LogLineResponse(
            level="L1",
            text="No ingestion run yet — click Refresh Live Data.",
            tab=0,
            source="real",
        )
    run_id = latest[0]["run_id"]
    connector_rows = execute_query(
        "SELECT rows_inserted, rows_skipped, duration_ms FROM ingestion_run_log WHERE run_id = ?",
        (run_id,),
    )
    total_inserted = sum(r["rows_inserted"] for r in connector_rows)
    total_skipped = sum(r["rows_skipped"] for r in connector_rows)
    return LogLineResponse(
        level="L1",
        text=(
            f"Ingested {total_inserted} rows ({total_skipped} skipped) across "
            f"{len(connector_rows)} connector(s) → run_id {run_id}"
        ),
        tab=0,
        source="real",
    )


@router.get("/logs", response_model=LiveFeedLogsResponse)
def get_live_feed_logs(run_id: str | None = None):
    """Return Agent Log lines from agent_execution_log when run_id is set and
    rows exist; otherwise one real L1 line + Day-1 L2-L7 fixture stubs."""
    if run_id:
        real_lines = fetch_run_logs(run_id)
        if real_lines:
            return LiveFeedLogsResponse(
                lines=[LogLineResponse(**line) for line in real_lines]
            )

    l1_line = _build_real_l1_log_line()
    stub_lines = [
        LogLineResponse(level=line["level"], text=line["text"], tab=line["tab"], source="stub")
        for line in _STUB_LOG_LINES_L2_L7
    ]
    return LiveFeedLogsResponse(lines=[l1_line, *stub_lines])


def _build_real_l1_gantt_bar() -> GanttBarResponse:
    """Build the one real Gantt bar for L1 from the latest ingestion_run_log
    batch — duration is the max single-connector duration_ms, i.e. the
    wall-clock span of the slowest connector in that batch (connectors run
    sequentially in DataIngestionAgent, so this approximates total L1 time
    without needing a separate batch-level start/end timestamp column)."""
    latest = execute_query(
        "SELECT run_id FROM ingestion_run_log ORDER BY run_ts_utc DESC LIMIT 1"
    )
    if not latest:
        return GanttBarResponse(id="L1", start=0, dur=0, color="#22C55E", source="real")
    run_id = latest[0]["run_id"]
    durations = execute_query(
        "SELECT duration_ms FROM ingestion_run_log WHERE run_id = ?", (run_id,)
    )
    total_ms = sum((r["duration_ms"] or 0) for r in durations)
    return GanttBarResponse(
        id="L1", start=0, dur=round(total_ms / 1000, 1), color="#22C55E", source="real"
    )


@router.get("/gantt", response_model=LiveFeedGanttResponse)
def get_live_feed_gantt(run_id: str | None = None):
    """Return Gantt bars from agent_execution_log when run_id is set and rows
    exist; otherwise one real L1 bar + Day-1 L2-L7 fixture stubs."""
    if run_id:
        real_bars = fetch_run_gantt(run_id)
        if real_bars:
            return LiveFeedGanttResponse(
                bars=[GanttBarResponse(**bar) for bar in real_bars]
            )

    l1_bar = _build_real_l1_gantt_bar()
    stub_bars = [
        GanttBarResponse(id=bar["id"], start=bar["start"], dur=bar["dur"], color=bar["color"], source="stub")
        for bar in _STUB_GANTT_BARS_L2_L7
    ]
    return LiveFeedGanttResponse(bars=[l1_bar, *stub_bars])
