"""
observability.py router — /api/observability/*

All four endpoints read from real SQLite tables (llm_call_log,
risk_classifications) populated during pipeline runs. No fixture fallback —
empty tables return [] so the dashboard shows an honest empty state.
Response shapes are identical to the Day-1 fixture contract.
"""

from typing import List

from fastapi import APIRouter

from src.api.schemas import CostByAgent, LatencyByAgent, PromptLogRow, VerdictDistributionSlice
from src.utils.db_utils import (
    fetch_cost_by_agent,
    fetch_latency_percentiles,
    fetch_prompt_log,
    fetch_verdict_distribution,
)

router = APIRouter()


@router.get("/cost", response_model=List[CostByAgent])
def get_cost():
    """Aggregate cost_usd per agent from llm_call_log. Was: FIXTURE COST_DATA."""
    return fetch_cost_by_agent()


@router.get("/verdicts", response_model=List[VerdictDistributionSlice])
def get_verdicts():
    """Verdict-type distribution from risk_classifications.full_result_json.

    Was: FIXTURE VERDICT_DIST."""
    return fetch_verdict_distribution()


@router.get("/latency", response_model=List[LatencyByAgent])
def get_latency():
    """P50/P90 latency per agent from llm_call_log (Python percentiles).

    Was: FIXTURE LATENCY_DATA."""
    return fetch_latency_percentiles()


@router.get("/prompt-log", response_model=List[PromptLogRow])
def get_prompt_log():
    """Latest LLM call rows from llm_call_log. Was: FIXTURE PROMPT_LOG."""
    rows = fetch_prompt_log(limit=50)
    return [PromptLogRow(**row) for row in rows]
