from fastapi import APIRouter
from src.api.schemas import MitigationResponse
from src.api.fixtures import MITIGATION

router = APIRouter()


@router.get("/{run_id}", response_model=MitigationResponse)
def get_mitigation(run_id: str):
    return {**MITIGATION, "run_id": run_id}
