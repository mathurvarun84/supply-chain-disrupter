from fastapi import APIRouter
from src.api.schemas import ForecastResponse
from src.api.fixtures import RUN_ID_FIXTURE, FORECAST_CATEGORIES, FORECAST_SERIES

router = APIRouter()


@router.get("/{run_id}", response_model=ForecastResponse)
def get_forecast(run_id: str, category: str = "Laptops"):
    return ForecastResponse(
        run_id=run_id,
        category=category,
        categories=FORECAST_CATEGORIES,
        series=FORECAST_SERIES,
    )
