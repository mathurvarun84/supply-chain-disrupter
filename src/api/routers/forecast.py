import json
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from src.api.fixtures import FORECAST_CATEGORIES, FORECAST_SERIES, RUN_ID_FIXTURE
from src.api.schemas import (
    ForecastResponse,
    ForecastWeekPoint,
    SkuForecastResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Path to pre-generated forecast JSONs produced by DemandForecastingAgent.run_all()
_FORECAST_OUTPUTS_DIR = Path(__file__).parents[3] / "data" / "forecast_outputs"


@router.get("/sku/list", response_model=List[str])
def list_forecast_skus():
    """Return all SKU IDs with persisted L5 forecast data.

    Checks the project SQLite DB first; falls back to pre-generated JSON
    files in data/forecast_outputs/ if the DB table is empty.
    """
    # Try DB first
    try:
        from src.utils.db_utils import list_forecast_skus as _db_list
        skus = _db_list()
        if skus:
            return skus
    except Exception as exc:
        logger.warning("DB list_forecast_skus failed: %s", exc)

    # Fallback: scan pre-generated JSON files
    if _FORECAST_OUTPUTS_DIR.exists():
        return sorted(
            p.stem.replace("forecast_result_", "")
            for p in _FORECAST_OUTPUTS_DIR.glob("forecast_result_*.json")
        )
    return []


@router.get("/sku/{sku_id}", response_model=SkuForecastResponse)
def get_sku_forecast(sku_id: str):
    """Return the 5-week Prophet forecast for a specific SKU (L5 DemandForecastingAgent v3).

    Lookup order:
      1. Project SQLite DB (demand_forecasts table) — written by the live pipeline.
      2. Pre-generated JSON files in data/forecast_outputs/ — capstone batch outputs.
    """
    # --- 1. Try DB ---
    try:
        from src.utils.db_utils import fetch_forecast_for_sku
        rows = fetch_forecast_for_sku(sku_id)
        if rows:
            week_points = [
                    ForecastWeekPoint(
                        week_start=r["week_start"],
                        demand_baseline=r["demand_baseline"],
                        demand_disrupted=r["demand_disrupted"],
                    )
                    for r in rows
                ]
            return SkuForecastResponse(
                sku_id=sku_id,
                forecast_horizon_weeks=len(rows),
                regressors_used=[],
                regressor_selection_method="backtest_ablation",
                expected_drop_pct=rows[0].get("deviation_pct") or 0.0,
                stockout_prob=rows[0].get("stockout_prob"),
                mape_prophet_trend_only=None,
                mape_prophet_selected=rows[0].get("mape_prophet"),
                mape_dataset_baseline_avg=None,
                mape_dataset_ai_avg=None,
                mape_improvement_pct_vs_dataset_baseline=None,
                disruption_scenario=None,
                demand_forecast=week_points,
                prophet_forecast=week_points,
                generated_at_utc=rows[0].get("generated_at_utc"),
            )
    except Exception as exc:
        logger.warning("DB fetch_forecast_for_sku failed for %s: %s", sku_id, exc)

    # --- 2. Fallback: pre-generated JSON ---
    json_path = _FORECAST_OUTPUTS_DIR / f"forecast_result_{sku_id}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for SKU '{sku_id}'. Run the L5 agent to generate one.",
        )

    with json_path.open() as fh:
        data = json.load(fh)

    raw_forecast = data.get("demand_forecast") or data.get("prophet_forecast", [])
    week_points = [
        ForecastWeekPoint(
            week_start=pt["week_start"],
            demand_baseline=pt["demand_baseline"],
            demand_disrupted=pt["demand_disrupted"],
        )
        for pt in raw_forecast
    ]
    return SkuForecastResponse(
        sku_id=data.get("sku_id", sku_id),
        forecast_horizon_weeks=len(raw_forecast),
        regressors_used=data.get("regressors_used", []),
        regressor_selection_method=data.get("regressor_selection_method", "backtest_ablation"),
        model_selected=data.get("model_selected", "prophet"),
        model_comparison_scores=data.get("model_comparison_scores", {}),
        expected_drop_pct=data.get("expected_drop_pct") or 0.0,
        stockout_prob=data.get("stockout_prob"),
        mape_prophet_trend_only=data.get("mape_prophet_trend_only"),
        mape_prophet_selected=data.get("mape_prophet_selected"),
        mape_dataset_baseline_avg=data.get("mape_dataset_baseline_avg"),
        mape_dataset_ai_avg=data.get("mape_dataset_ai_avg"),
        mape_improvement_pct_vs_dataset_baseline=data.get("mape_improvement_pct_vs_dataset_baseline"),
        disruption_scenario=data.get("disruption_scenario"),
        demand_forecast=week_points,
        prophet_forecast=week_points,
        generated_at_utc=None,
    )


@router.get("/{run_id}", response_model=ForecastResponse)
def get_forecast(run_id: str, category: str = "Laptops"):
    """Legacy category-level fixture endpoint (retained for dashboard compatibility).

    Defined last so it does not shadow the static /sku/* routes above.
    """
    return ForecastResponse(
        run_id=run_id,
        category=category,
        categories=FORECAST_CATEGORIES,
        series=FORECAST_SERIES,
    )
