from typing import List
from fastapi import APIRouter
from src.api.schemas import GuardrailEvent
from src.api.fixtures import GUARDRAIL_TABLE

router = APIRouter()


@router.get("/events", response_model=List[GuardrailEvent])
def get_guardrail_events():
    return GUARDRAIL_TABLE
