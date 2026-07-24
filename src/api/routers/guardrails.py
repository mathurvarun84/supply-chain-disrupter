"""Guardrails (Screen 5) — reads guardrail_events, the event log written by
src.utils.guardrails.log_guardrail_event() at every input/output/execution
checkpoint (see that module for the 16 guardrail functions)."""

from typing import List, Optional

from fastapi import APIRouter, Query

from src.api.schemas import GuardrailEvent
from src.utils.db_utils import count_slack_suppressed_by_guardrail, fetch_guardrail_events

router = APIRouter()


@router.get("/events", response_model=List[GuardrailEvent])
def get_guardrail_events(
    agent_name: Optional[str] = Query(None),
    guardrail_name: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    passed: Optional[bool] = Query(None),
):
    """Aggregated pass/fail counts per (guardrail_name, direction, agent_name).

    Consumed by: Screen 5 Guardrails sub-tab's Activity table."""
    return fetch_guardrail_events(
        agent_name=agent_name, guardrail_name=guardrail_name,
        direction=direction, passed=passed,
    )


@router.get("/slack-suppressed-count")
def get_slack_suppressed_count():
    """Doc §6 headline metric. Consumed by: Guardrails sub-tab's
    'Slack Alerts Suppressed by Guardrail' counter."""
    return {"count": count_slack_suppressed_by_guardrail()}
