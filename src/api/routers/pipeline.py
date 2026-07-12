"""
Pipeline control (top status bar) — POST /run starts a real L1-L7 pipeline
run as a FastAPI BackgroundTask and GET /status reports its live progress.

Deliberate choice: the BackgroundTask target is run_agent_sequence(), NOT
run_agent_graph() (the compiled LangGraph StateGraph). run_agent_sequence()
wraps every agent with agent_span()/pipeline_trace() (src/utils/
observability.py), which is what makes agent_execution_log — and therefore
this router's GET /status — non-empty. run_agent_graph() has no such
instrumentation (confirmed by reading it: no pipeline_trace, no agent_span,
GlobalState() built with no run_id). Do NOT "helpfully" switch this to
run_agent_graph() without first adding equivalent observability there.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.agents.data_ingestion_agent import DataIngestionAgent
from src.agents.demo_injector import build_demo_payload
from src.agents.langgraph_engine import run_agent_sequence
from src.agents.pipeline_bridge import snapshot_run_outputs
from src.api.schemas import AgentState, PipelineRunRequest, PipelineRunResponse, PipelineStatus
from src.utils.db_utils import (
    build_idle_agents,
    fetch_pipeline_status,
    fetch_recent_completed_run_ids,
    fetch_scenario_options,
)
from src.utils.observability import build_langfuse_trace_url

logger = logging.getLogger(__name__)
router = APIRouter()

_MODE_TO_SOURCE_TYPE = {"live": "LIVE", "demo": "DEMO-INJECTED", "replay": "REPLAY"}

# In-process only: maps run_id -> source_type for runs triggered via POST
# /run this server process. agent_execution_log has no mode/source_type
# column, so a run_id's source_type can't be recovered after a restart —
# _infer_source_type() below falls back to "REPLAY" in that case.
_RUN_SOURCE_TYPE: Dict[str, str] = {}

# In-process only: run_id -> human-readable phase text for the window
# BEFORE L1's agent_span() writes the first agent_execution_log row (live
# mode's connector sweep can take 1-3 minutes). Without this, GET /status
# 404s during that window and the frontend has nothing to show. Cleared
# once the ingestion sweep finishes, since agent_execution_log takes over
# as the source of truth from L1 onward.
_RUN_PHASE: Dict[str, str] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_live_payload(run_id: str) -> Dict[str, Any]:
    """Live-mode payload: same EventMetadata shape the Streamlit dashboard's
    manual scenario form and scripts/seed_demo_run.py already use (a real
    (port, sku, event_date) baseline from daily_records) — mode="live" only
    changes what data_ingestion_agent_v2 prefers (live enrichment overlay
    over the historical record), not the payload shape itself."""
    options = fetch_scenario_options()
    if not options:
        raise HTTPException(
            status_code=503,
            detail="No scenario baselines available — run scripts/build_databases.py",
        )
    scenario = max(options, key=lambda r: r.get("history_points") or 0)
    return {
        "run_id": run_id,
        "mode": "live",
        "source_type": "LIVE",
        "disruption_type": "geopolitical",
        "affected_port": scenario["port"],
        "affected_route": f"{scenario['port']} to Singapore",
        "severity": 0.6,
        "shock_duration_days": 14,
        "recovery_window_days": 90,
        "synthetic_ratio": 0.0,
        "event_date": scenario["event_date"],
        "sku": scenario["sku"],
    }


def _build_payload(request: PipelineRunRequest, run_id: str) -> Dict[str, Any]:
    if request.mode == "demo":
        return build_demo_payload(request.demo_scenario_id, run_id)
    return _build_live_payload(run_id)


def _refresh_live_data() -> None:
    """Run one DataIngestionAgent connector sweep (Open-Meteo, GDELT, FRED,
    Google News/Reuters RSS, CISA/BIS, YFinance) synchronously, writing
    fresh rows to live_news_ingest / live_weather_ingest / live_enrichment
    before L1 runs. Same call the Live Feed tab's "Refresh Live Data"
    button makes (POST /api/live-feed/refresh -> _run_ingestion_batch());
    self-guards via DataIngestionAgent's own _INGESTION_LOCK, so this is a
    no-op (logged, not raised) if a manual refresh is already in flight.
    Runs inside the BackgroundTask, off the request thread, so blocking
    here for the ~10-60s a full sweep takes is fine."""
    result = DataIngestionAgent().run_batch()
    if result.status == "skipped":
        logger.warning(
            "Live pipeline run: ingestion batch skipped (already running), "
            "proceeding with whatever live_enrichment already exists."
        )


def _run_and_snapshot(run_id: str, payload: Dict[str, Any]) -> None:
    """BackgroundTask body: run the real pipeline, then bridge L4/L6/L7's
    output into the dashboard tables. Per-agent failure handling already
    lives inside run_agent_sequence() (L1-L4/L7 critical, L5/L6 optional);
    this wrapper only needs to snapshot whatever GlobalState comes back.

    Live mode additionally fetches fresh news/weather first (see
    _refresh_live_data) so (a) the Live Feed tab actually has new data to
    show after a live run, and (b) L1's live-enrichment overlay reflects
    this run, not whatever a previous, unrelated batch last wrote."""
    if payload.get("mode") == "live":
        _RUN_PHASE[run_id] = "Fetching live news & weather data…"
        try:
            _refresh_live_data()
        finally:
            _RUN_PHASE.pop(run_id, None)
    final_state = run_agent_sequence(payload)
    snapshot_run_outputs(run_id, final_state)


@router.get("/status", response_model=PipelineStatus)
def get_status(run_id: Optional[str] = None) -> PipelineStatus:
    """Reads agent_execution_log (written by agent_span()) for run_id,
    reshaped into the 7-entry Idle/Running/Complete/Skipped-Optional/
    Failed-Fallback contract. When run_id is omitted, reports the most
    recently active run so the status bar has something to show on first
    load, before any run has been explicitly triggered."""
    status = fetch_pipeline_status(run_id)
    if status is None:
        phase = _RUN_PHASE.get(run_id) if run_id else None
        if run_id is None or phase is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown run_id={run_id}" if run_id else "No pipeline runs yet.",
            )
        # Live mode's pre-L1 window: this run_id was accepted by POST /run
        # but its connector sweep hasn't finished, so agent_execution_log
        # has no rows for it yet. Report all-Idle + current_phase instead
        # of 404ing, so the status bar has something to show immediately.
        return PipelineStatus(
            run_id=run_id,
            source_type=_infer_source_type(run_id),
            agents=[AgentState(**agent) for agent in build_idle_agents()],
            last_ingested_at=None,
            openai_status="connected",
            langfuse_trace_url=build_langfuse_trace_url(run_id),
            is_complete=False,
            current_phase=phase,
        )
    return PipelineStatus(
        run_id=status["run_id"],
        source_type=_infer_source_type(status["run_id"]),
        agents=[AgentState(**agent) for agent in status["agents"]],
        last_ingested_at=status["last_ingested_at"],
        openai_status="connected",
        langfuse_trace_url=build_langfuse_trace_url(status["run_id"]),
        is_complete=status["is_complete"],
        current_phase=_RUN_PHASE.get(status["run_id"]),
    )


def _infer_source_type(run_id: str) -> str:
    """PipelineStatus.source_type is display-only metadata. Runs triggered
    through this process's own POST /run are looked up in _RUN_SOURCE_TYPE;
    a run_id from before this process started (or after a restart) falls
    back to REPLAY, the conservative default (scripts/seed_demo_run.py's
    own payload also uses mode="replay"/source_type="REPLAY")."""
    return _RUN_SOURCE_TYPE.get(run_id, "REPLAY")


@router.get("/runs")
def list_recent_runs(limit: int = 10) -> Dict[str, Any]:
    """Recent completed run_ids for the Demo Scenario Injector's Replay
    picker (a plain <select>, per the scope cut — not a searchable UI)."""
    return {"runs": fetch_recent_completed_run_ids(limit)}


@router.post("/run", response_model=PipelineRunResponse)
def run_pipeline(
    request: PipelineRunRequest, background_tasks: BackgroundTasks
) -> PipelineRunResponse:
    if request.mode == "demo" and not request.demo_scenario_id:
        raise HTTPException(status_code=422, detail="demo_scenario_id is required when mode='demo'")
    if request.mode == "replay" and not request.replay_run_id:
        raise HTTPException(status_code=422, detail="replay_run_id is required when mode='replay'")

    if request.mode == "replay":
        existing = fetch_pipeline_status(request.replay_run_id)
        if existing is None or not existing["is_complete"]:
            raise HTTPException(
                status_code=404,
                detail=f"No completed run for replay_run_id={request.replay_run_id}",
            )
        _RUN_SOURCE_TYPE[request.replay_run_id] = "REPLAY"
        return PipelineRunResponse(
            run_id=request.replay_run_id, mode="replay", accepted_at=_utcnow_iso()
        )

    run_id = str(uuid.uuid4())
    payload = _build_payload(request, run_id)
    _RUN_SOURCE_TYPE[run_id] = _MODE_TO_SOURCE_TYPE[request.mode]
    if request.mode == "live":
        # Set immediately (not just inside the BackgroundTask) so GET
        # /status never sees a gap between accept and the task actually
        # starting to execute.
        _RUN_PHASE[run_id] = "Fetching live news & weather data…"
    background_tasks.add_task(_run_and_snapshot, run_id, payload)
    return PipelineRunResponse(run_id=run_id, mode=request.mode, accepted_at=_utcnow_iso())
