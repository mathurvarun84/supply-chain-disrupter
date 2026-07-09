"""Tests for L6 Monte Carlo simulation agent."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from src.agents.simulation_agent.agent import simulation_agent
from src.agents.simulation_agent.engine import run_monte_carlo
from src.agents.simulation_agent.priors import SimulationParams, build_simulation_params, resolve_alternate_route
from src.agents.state import EventMetadata, ForecastResult, GlobalState, RiskClassificationResult
from src.utils.db_utils import DB_PATH, ensure_simulation_schema


def _base_params(**overrides) -> SimulationParams:
    defaults = dict(
        initial_inventory=200.0,
        incoming_supply=100.0,
        baseline_lead_time=7.0,
        mean_daily_demand=50.0,
        unit_price_usd=10.0,
        horizon_days=30,
        severity=0.5,
        shock_duration_days=5,
        disruption_type="port closure",
        composite_score=0.5,
        supply_disruption_index=6.0,
        defect_rate_pct=5.0,
        alternate_route="Suez Canal",
        logistics_disruption=True,
        trials=500,
        seed=42,
    )
    defaults.update(overrides)
    return SimulationParams(**defaults)


def _risk_state(**kwargs) -> GlobalState:
    active_extra = kwargs.pop("active_record", {})
    sim_trials = kwargs.pop("simulation_trials", 500)
    risk = RiskClassificationResult(
        mode="live",
        composite_score=kwargs.pop("composite_score", 0.55),
        geo_component=0.4,
        supply_component=0.5,
        freight_component=0.6,
        defect_component=0.4,
        duration_days=3.0,
        base_label="HIGH",
        final_label="HIGH",
        escalated=False,
        rationale="test",
        critical_flag=False,
    )
    return GlobalState(
        event_metadata=EventMetadata(
            disruption_type=kwargs.pop("disruption_type", "port closure"),
            affected_port="Eastern Asia",
            affected_route="test",
            severity=kwargs.pop("severity", 0.6),
            shock_duration_days=kwargs.pop("shock_duration_days", 5),
            recovery_window_days=kwargs.pop("recovery_window_days", 30),
            synthetic_ratio=0.0,
            simulation_trials=sim_trials,
        ),
        config={
            "route_maps": {"JNPT": {"backup_route": "Cape of Good Hope"}},
            "region_route_maps": {"Eastern Asia": {"backup_route": "Suez Canal"}},
        },
        active_record={
            "event_date": "2024-01-01",
            "port": "Eastern Asia",
            "order_region": "Eastern Asia",
            "sku": "CHIP_AP",
            "inventory_level": 200.0,
            "incoming_supply": 100.0,
            "lead_time_days": 7.0,
            "demand": 50.0,
            "sales_usd": 500.0,
            "unit_price_usd": 10.0,
            "supply_disruption_index": 6.5,
            "defect_rate_pct": 5.0,
            **active_extra,
        },
        risk_classification=risk,
        **kwargs,
    )


def test_deterministic_seed():
  params = _base_params(seed=123, trials=300)
  r1 = run_monte_carlo(params)
  r2 = run_monte_carlo(params)
  assert r1.stockout_probability_pct == r2.stockout_probability_pct
  assert r1.revenue_impact_usd_p50 == r2.revenue_impact_usd_p50


def test_high_severity_wider_tail():
  low = run_monte_carlo(_base_params(severity=0.1, shock_duration_days=1, seed=7))
  high = run_monte_carlo(_base_params(severity=0.95, shock_duration_days=14, seed=7))
  assert high.stockout_probability_p90 >= low.stockout_probability_p90


def test_l5_forecast_shifts_revenue_distribution():
  demands = [40.0] * 30
  with_forecast = _base_params(forecast_daily_demands=demands, seed=11)
  without_forecast = _base_params(forecast_daily_demands=[], mean_daily_demand=80.0, seed=11)
  r_forecast = run_monte_carlo(with_forecast)
  r_no_forecast = run_monte_carlo(without_forecast)
  assert r_no_forecast.revenue_impact_usd_p50 >= r_forecast.revenue_impact_usd_p50


def test_resolve_alternate_route_by_region():
  config = {"region_route_maps": {"Eastern Asia": {"backup_route": "Suez Canal"}}}
  route = resolve_alternate_route(config, {"port": "Eastern Asia", "order_region": "Eastern Asia"})
  assert route == "Suez Canal"


def test_simulation_agent_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", tmp_path / "test.db")

    state = _risk_state(simulation_trials=200)
    with patch("src.agents.simulation_agent.agent.insert_simulation_run") as mock_insert:
        result = simulation_agent(state)
    assert result["simulation_result"].trials_run == 200
    assert result["simulation_result"].stockout_probability_pct >= 0.0
    mock_insert.assert_called_once()


def test_simulation_schema_insert(tmp_path, monkeypatch):
  db_path = tmp_path / "sim.db"
  monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

  from src.agents.state import SimulationResult
  from src.utils.db_utils import insert_simulation_run

  sim = SimulationResult(
      stockout_probability_pct=45.0,
      expected_inventory_gap_pct=30.0,
      alternate_route="Suez Canal",
      stockout_probability_p10=20.0,
      stockout_probability_p90=70.0,
      revenue_impact_usd_p50=1000.0,
      revenue_impact_usd_p10=500.0,
      revenue_impact_usd_p90=2000.0,
      days_to_stockout_p50=10.0,
      trials_run=500,
      model_version="mc_v1",
  )
  insert_simulation_run(
      "2024-01-01", "Eastern Asia", "CHIP_AP", "earthquake", "HIGH", sim, "{}"
  )

  conn = sqlite3.connect(db_path)
  row = conn.execute("SELECT stockout_p50, trials_run FROM simulation_runs").fetchone()
  conn.close()
  assert row[0] == 45.0
  assert row[1] == 500


def test_build_simulation_params_from_state():
  state = _risk_state(
      forecast_result=ForecastResult(
          prophet_forecast=[{"ds": "2024-01-01", "yhat": 45.0}] * 30,
          expected_drop_pct=10.0,
      )
  )
  with patch("src.agents.simulation_agent.priors.fetch_ops_kpi_priors", return_value=None):
    params = build_simulation_params(state, trials=100, seed=1)
  assert params.alternate_route == "Suez Canal"
  assert len(params.forecast_daily_demands) == 30


def test_simulation_trials_from_event_metadata():
    state = _risk_state(simulation_trials=350)
    trials = __import__(
        "src.agents.simulation_agent.agent", fromlist=["_trial_count"]
    )._trial_count(state)
    assert trials == 350


def test_simulation_agent_missing_record_raises():
  state = GlobalState(config={"route_maps": {}}, active_record=None)
  with pytest.raises(ValueError, match="Active record"):
    simulation_agent(state)
