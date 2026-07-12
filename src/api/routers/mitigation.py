"""Mitigation Plan (Screen 4) — reads mitigation_output for a pipeline run_id."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import MitigationResponse
from src.utils.db_utils import fetch_mitigation

router = APIRouter()


@router.get("/{run_id}", response_model=MitigationResponse)
def get_mitigation(run_id: str):
    """Reads mitigation_output (+ RAG trace JSON) for run_id.

    Was: return FIXTURE_MITIGATION from src/api/fixtures.py.
    Consumed by: Screen 4 Mitigation tab."""
    result = fetch_mitigation(run_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No mitigation plan for run_id={run_id}",
        )
    return MitigationResponse(**result)
