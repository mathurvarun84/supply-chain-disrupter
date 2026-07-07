from fastapi import APIRouter
from src.api.schemas import PipelineStatus, PipelineRunRequest, PipelineRunResponse
from src.api.fixtures import INITIAL_AGENTS, RUN_ID_FIXTURE

router = APIRouter()


@router.get("/status", response_model=PipelineStatus)
def get_status():
    return PipelineStatus(
        run_id=RUN_ID_FIXTURE,
        source_type="REPLAY",
        agents=INITIAL_AGENTS,
        last_ingested_at=None,
        openai_status="connected",
    )


@router.post("/run", response_model=PipelineRunResponse)
def run_pipeline(body: PipelineRunRequest):
    # Day 9 wires this to a real BackgroundTask that drives run_agent_graph().
    return PipelineRunResponse(run_id=RUN_ID_FIXTURE)
