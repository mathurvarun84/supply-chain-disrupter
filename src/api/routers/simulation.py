from fastapi import APIRouter
from src.api.schemas import SimulationResponse
from src.api.fixtures import MONTE_CARLO

router = APIRouter()


@router.get("/{run_id}", response_model=SimulationResponse)
def get_simulation(run_id: str):
    return SimulationResponse(
        run_id=run_id,
        p10=18.0,
        p50=41.0,
        p90=68.0,
        revenue_at_risk_usd=4200000,
        alternate_route="Cape of Good Hope",
        histogram=MONTE_CARLO,
    )
