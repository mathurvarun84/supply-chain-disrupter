from typing import List
from fastapi import APIRouter
from src.api.schemas import CostByAgent, VerdictDistributionSlice, LatencyByAgent, PromptLogRow
from src.api.fixtures import COST_DATA, VERDICT_DIST, LATENCY_DATA, PROMPT_LOG

router = APIRouter()


@router.get("/cost", response_model=List[CostByAgent])
def get_cost():
    return COST_DATA


@router.get("/verdicts", response_model=List[VerdictDistributionSlice])
def get_verdicts():
    return VERDICT_DIST


@router.get("/latency", response_model=List[LatencyByAgent])
def get_latency():
    return LATENCY_DATA


@router.get("/prompt-log", response_model=List[PromptLogRow])
def get_prompt_log():
    return PROMPT_LOG
