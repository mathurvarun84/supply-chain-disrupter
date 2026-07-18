"""
trulens.py — POST /api/trulens/run triggers one demo-scenario pipeline run
instrumented with TruLens (src.evaluation.trulens_integration.wrapper.
run_with_trulens), completely separate from the live "Run Pipeline" button —
see wrapper.py's module docstring for why it drives build_agent_graph()
directly instead of run_pipeline(). GET /status/{run_id} polls the
in-process result (mirrors src/api/routers/pipeline.py's _RUN_PHASE pattern;
TruLens runs don't write agent_execution_log so there's no DB-backed status
to read). GET /metrics reports the one domain feedback function
(risk_score_stability) that has a persisted historical data source outside
a single run — see feedback_functions.py's docstring for why the other
three (ensemble_agreement, node_latency_check, forecast_accuracy) aren't
surfaced here without a completed run to compute them from.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.agents.demo_injector import build_demo_payload
from src.api.schemas import DemoScenarioId
from src.evaluation.trulens_integration.feedback_functions import (
    node_latency_check,
    risk_score_stability,
)
from src.evaluation.trulens_integration.wrapper import run_with_trulens
from src.utils.db_utils import fetch_recent_composite_scores

logger = logging.getLogger(__name__)
router = APIRouter()

# In-process only, same tradeoff as pipeline.py's _RUN_PHASE/_RUN_SOURCE_TYPE:
# a capture run started before a server restart is unrecoverable, which is
# fine here since this is a manual dev/demo/grading tool, not a durable record.
_RUNS: Dict[str, Dict[str, Any]] = {}


class TruLensRunRequest(BaseModel):
    demo_scenario_id: DemoScenarioId


class TruLensRunResponse(BaseModel):
    run_id: str
    accepted_at: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execute(run_id: str, demo_scenario_id: str) -> None:
    _RUNS[run_id] = {"status": "running", "demo_scenario_id": demo_scenario_id}
    try:
        payload = build_demo_payload(demo_scenario_id, run_id)
        capture: Dict[str, Any] = {}
        final_state = run_with_trulens(payload, capture=capture)
        node_latencies = capture.get("node_latencies_ms", {})
        cost_summary = capture.get("cost_summary", {})
        _RUNS[run_id] = {
            "status": "complete",
            "demo_scenario_id": demo_scenario_id,
            "risk_label": final_state.risk_label,
            "node_latencies_ms": node_latencies,
            "cost_summary": cost_summary,
            "node_latency_check": round(node_latency_check(node_latencies), 3),
            "completed_at": _utcnow_iso(),
        }
    except Exception as exc:
        logger.exception("TruLens capture failed for run_id=%s", run_id)
        _RUNS[run_id] = {
            "status": "failed",
            "demo_scenario_id": demo_scenario_id,
            "error": str(exc),
        }


@router.post("/run", response_model=TruLensRunResponse)
def submit_trulens_run(
    request: TruLensRunRequest, background_tasks: BackgroundTasks
) -> TruLensRunResponse:
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {"status": "pending", "demo_scenario_id": request.demo_scenario_id}
    background_tasks.add_task(_execute, run_id, request.demo_scenario_id)
    return TruLensRunResponse(run_id=run_id, accepted_at=_utcnow_iso())


@router.get("/status/{run_id}")
def get_trulens_status(run_id: str) -> Dict[str, Any]:
    result = _RUNS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown TruLens run_id={run_id}")
    return {"run_id": run_id, **result}


@router.get("/metrics")
def get_trulens_metrics(days: int = 30) -> Dict[str, Any]:
    scores = fetch_recent_composite_scores(days)
    return {
        "days": days,
        "n_runs": len(scores),
        "risk_score_stability": round(risk_score_stability(scores), 3),
    }
