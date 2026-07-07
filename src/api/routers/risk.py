from fastapi import APIRouter
from src.api.schemas import RiskClassification
from src.api.fixtures import RISK_CLASSIFICATION

router = APIRouter()


@router.get("/{run_id}", response_model=RiskClassification)
def get_risk_classification(run_id: str):
    return {**RISK_CLASSIFICATION, "run_id": run_id}
