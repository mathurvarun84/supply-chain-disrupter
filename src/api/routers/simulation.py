"""Forecast & Simulation (Screen 3, Simulation tab) — reads simulation_output
for a pipeline run_id."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import SimulationResponse
from src.utils.db_utils import fetch_simulation

router = APIRouter()


@router.get("/{run_id}", response_model=SimulationResponse)
def get_simulation(run_id: str) -> SimulationResponse:
    """Reads simulation_output for run_id.

    Was: hardcoded fixture values regardless of run_id.
    Consumed by: Screen 3 Simulation tab."""
    result = fetch_simulation(run_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation result for run_id={run_id}",
        )
    return SimulationResponse(**result)
