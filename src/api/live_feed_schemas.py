"""
Pydantic response models for Screen 1 (Live Feed) — L1 ingestion endpoints only.

Consumed by src/api/routers/live_feed.py and mirrored 1:1 in the frontend at
src/frontend/src/app/types/liveFeed.ts. Keep both in sync when changing fields.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NewsHeadline(BaseModel):
    """One row from live_news_ingest for the News panel.

    score_tier is server-computed (never in React) so the low/medium/high
    threshold logic lives in exactly one place and can't drift between screens.
    """

    hub_city: Optional[str] = None
    hub_country: Optional[str] = None
    supplier_country: Optional[str] = None
    headline: str
    published_at: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    query_term: Optional[str] = None
    source_feed: Optional[str] = None
    score_tier: Literal["low", "medium", "high"] = "low"


class LiveFeedNewsResponse(BaseModel):
    """GET /api/live-feed/news payload. Empty items list (not a 404/500) when
    no ingestion run has happened yet, so the frontend can render a clear
    'no data yet' state."""

    run_id: Optional[str]
    count: int
    fetched_at: Optional[str]
    items: List[NewsHeadline]


class WeatherHub(BaseModel):
    """One fab-hub weather card. The endpoint always returns all 6 configured
    hubs (from HUB_CITIES) even if a hub has no ingested row yet — a missing
    hub becomes an explicit 'no data yet' card, never a silently dropped one."""

    hub_city: str
    wind_speed_kmh: Optional[float] = None
    precipitation_mm: Optional[float] = None
    weather_code: Optional[int] = None
    temperature_c: Optional[float] = None
    raw_severity_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    is_trigger_hub: bool = False
    fetched_at_utc: Optional[str] = None


class LiveFeedWeatherResponse(BaseModel):
    """GET /api/live-feed/weather payload. `hubs` always has exactly 6 entries."""

    run_id: Optional[str]
    fetched_at: Optional[str]
    hubs: List[WeatherHub]


class IngestRunLogRow(BaseModel):
    """One row of ingestion_run_log — one connector's result within a batch run."""

    run_ts_utc: str
    source: str
    status: str
    rows_inserted: int
    rows_skipped: int
    duration_ms: Optional[int]
    error_detail: Optional[str]


class IngestStatusResponse(BaseModel):
    """GET /api/live-feed/ingest-status payload. `is_running` reflects
    DataIngestionAgent's own module-level threading lock — never a second,
    independently-tracked flag, to avoid two sources of truth."""

    is_running: bool
    last_run: Optional[IngestRunLogRow]
    recent_runs: List[IngestRunLogRow]


class RefreshTriggeredResponse(BaseModel):
    """POST /api/live-feed/refresh payload. 'skipped_already_running' means a
    batch was already in flight and this call did not queue a second one."""

    status: Literal["started", "skipped_already_running"]
    triggered_at: str


class LogLineResponse(BaseModel):
    """One Agent Log line. `source` distinguishes the real L1 line (built
    from ingestion_run_log) from the unmodified Day-1 L2-L7 fixture lines,
    so the frontend can render stub entries visibly dimmed rather than
    hiding the scope cut."""

    level: str
    text: str
    tab: int
    source: Literal["real", "stub"]


class LiveFeedLogsResponse(BaseModel):
    lines: List[LogLineResponse]


class GanttBarResponse(BaseModel):
    """One Gantt strip bar. `source` mirrors LogLineResponse — real L1 bar
    computed from ingestion_run_log durations, stub L2-L7 bars unchanged
    from the Day-1 fixture."""

    id: str
    start: float
    dur: float
    color: str
    source: Literal["real", "stub"]


class LiveFeedGanttResponse(BaseModel):
    bars: List[GanttBarResponse]
