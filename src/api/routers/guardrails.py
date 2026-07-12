"""Guardrails (Screen 5) — reads guardrail_events aggregate table."""

from typing import List

from fastapi import APIRouter

from src.api.schemas import GuardrailEvent
from src.utils.db_utils import fetch_guardrail_events

router = APIRouter()


@router.get("/events", response_model=List[GuardrailEvent])
def get_guardrail_events():
    """Reads guardrail_events (input + output guardrail pass/fail aggregates).

    Was: return FIXTURE_GUARDRAIL_TABLE from src/api/fixtures.py.
    Consumed by: Screen 5 Guardrails sub-tab."""
    return fetch_guardrail_events()
