import importlib.util
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils.db_utils import (
    fetch_time_series,
    insert_mitigation_action,
)
from src.utils.yaml_utils import get_route_map
from src.agents.data_ingestion.agent import data_ingestion_agent
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.risk_classifier_agent import risk_classifier_agent
from src.agents.state import (
    ForecastResult,
    GlobalState,
    MitigationAction,
    SimulationResult,
)

logger = logging.getLogger(__name__)

# Optional heavy dependencies — agents that need these degrade gracefully when absent.
_PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None
_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class MitigationSchema(BaseModel):
    summary: str = Field(...)
    recommendations: List[str] = Field(..., min_items=3, max_items=3)
    cost_delta: str = Field(...)


def demand_forecasting_agent(state: GlobalState) -> Dict[str, Any]:
    if not _PROPHET_AVAILABLE or not _PANDAS_AVAILABLE:
        logger.warning("L5: prophet/pandas not installed — demand forecasting skipped.")
        return {
            "agent_logs": state.agent_logs + [
                "L5: SKIPPED — prophet or pandas not installed. "
                "Run: pip install prophet pandas"
            ],
        }

    if state.active_record is None:
        raise ValueError("Active record is required for demand forecasting.")

    ts = fetch_time_series(state.active_record["port"], state.active_record["sku"])
    if len(ts) < 10:
        return {
            "agent_logs": state.agent_logs + [
                f"L5: SKIPPED — only {len(ts)} history points available (need ≥ 10)."
            ],
        }

    import pandas as pd
    from prophet import Prophet

    df_records = [{"ds": row["event_date"], "y": row["demand"]} for row in ts]
    df = pd.DataFrame(df_records)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast_points = forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")
    demand_baseline = float(state.active_record.get("demand", 0.0))
    expected_drop = max(0.0, 1.0 - (forecast_points[-1]["yhat"] / (demand_baseline or 1.0)))
    return {
        "forecast_result": ForecastResult(
            prophet_forecast=forecast_points,
            expected_drop_pct=round(expected_drop * 100.0, 2),
        ),
        "agent_logs": state.agent_logs + ["L5: Demand forecasting completed."],
    }


def simulation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.active_record is None or state.config is None:
        raise ValueError("Active record and config are required for simulation.")
    current_inventory = float(state.active_record.get("inventory_level", 0.0))
    incoming = float(state.active_record.get("incoming_supply", 0.0))
    lead_time = float(state.active_record.get("lead_time_days", 1.0))
    alt_route = get_route_map(state.config, state.active_record["port"]).get("backup_route", "Cape of Good Hope")
    stockout_probability = min(100.0, max(0.0, (state.risk_score_composite or 0.0) * 100.0 + (1.0 - (current_inventory / (incoming + 1.0))) * 25.0 + (lead_time / 30.0) * 25.0))
    expected_gap = max(0.0, 100.0 - (current_inventory / (incoming + 1.0)) * 100.0)
    return {
        "simulation_result": SimulationResult(
            stockout_probability_pct=round(stockout_probability, 2),
            expected_inventory_gap_pct=round(expected_gap, 2),
            alternate_route=alt_route,
        ),
        "agent_logs": state.agent_logs + ["L6: Simulation completed."],
    }


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.risk_label is None:
        raise ValueError("Risk label is required for mitigation — run risk_classifier_agent first.")

    # Simulation and forecast are optional — use fallback values when not available.
    stockout = state.simulation_result.stockout_probability_pct if state.simulation_result else None
    forecast_drop = state.forecast_result.expected_drop_pct if state.forecast_result else None
    alt_route = (
        state.simulation_result.alternate_route
        if state.simulation_result
        else "the configured backup route"
    ) or "the configured backup route"

    stockout_note = f"{stockout:.1f}%" if stockout is not None else "unknown (simulation not run)"
    forecast_note = f"{forecast_drop:.1f}%" if forecast_drop is not None else "unknown (forecast not run)"

    recommendations = [
        f"Raise safety stock for the affected product — stockout estimate: {stockout_note}.",
        f"Prepare diversion through {alt_route} and confirm carrier capacity.",
        f"Review alternate suppliers and align purchase orders to forecast variance: {forecast_note}.",
    ]
    cost_delta = (
        "High: expedite critical inventory and activate alternate sourcing."
        if state.risk_label == "CRITICAL"
        else "Moderate: reserve backup logistics and inventory capacity."
    )
    parsed = MitigationSchema(
        summary=(
            f"{state.risk_label} electronics supply-chain risk requires "
            "inventory, routing, and supplier actions."
        ),
        recommendations=recommendations,
        cost_delta=cost_delta,
    )
    insert_mitigation_action(
        state.active_record["event_date"],
        state.active_record["port"],
        state.active_record["sku"],
        state.risk_label,
        json.dumps(parsed.recommendations),
        parsed.cost_delta,
    )
    # Slack trigger: hard business rule — fire when critical_flag is set
    if state.risk_classification and state.risk_classification.critical_flag:
        # fire Slack webhook here — this is the hard business rule
        pass
    return {
        "mitigation_action": MitigationAction(**parsed.dict()),
        "agent_logs": state.agent_logs + ["L7: Mitigation recommendation generated and persisted."],
    }


def _run_optional(
    state: GlobalState,
    agent_fn,
    label: str,
) -> GlobalState:
    """
    Run an optional agent. On any exception, append a SKIPPED log entry and
    return the unchanged state so downstream agents can still run.
    """
    try:
        delta = agent_fn(state)
        return state.copy(update=delta)
    except Exception as exc:
        logger.warning("%s skipped: %s", label, exc)
        return state.copy(
            update={"agent_logs": state.agent_logs + [f"{label}: SKIPPED — {exc}"]}
        )


def run_agent_graph(payload: Dict[str, Any]) -> GlobalState:
    # ── Critical agents — raise on failure ───────────────────────────────────
    state = GlobalState()

    ingestion_delta = data_ingestion_agent(state, payload)
    state = state.copy(update=ingestion_delta)

    news_delta = news_event_analysis_agent(state)
    state = state.copy(update=news_delta)

    weather_delta = weather_risk_monitoring_agent(state)
    state = state.copy(update=weather_delta)

    risk_delta = risk_classifier_agent(state)
    state = state.copy(update=risk_delta)

    # ── Optional agents — log and continue on failure ─────────────────────────
    state = _run_optional(state, demand_forecasting_agent, "L5")
    state = _run_optional(state, simulation_agent, "L6")
    state = _run_optional(state, mitigation_recommendation_agent, "L7")

    return state
