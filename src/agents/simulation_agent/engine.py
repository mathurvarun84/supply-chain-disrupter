"""Monte Carlo discrete-time inventory simulation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.agents.simulation_agent.priors import SimulationParams
from src.agents.state import SimulationResult

MODEL_VERSION = "mc_v1"
MAX_HISTOGRAM_SAMPLES = 100


def _percentile(values: np.ndarray, q: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _sample_disruption_duration(params: SimulationParams, rng: np.random.Generator) -> int:
    if params.shock_duration_days > 0:
        jitter = rng.integers(-1, 2)
        return max(1, params.shock_duration_days + int(jitter))
    if params.expected_duration_days is not None and params.expected_duration_days > 0:
        return max(1, int(rng.lognormal(np.log(params.expected_duration_days), 0.35)))
    base = 3 + int(params.severity * 14)
    return max(1, int(rng.normal(base, max(1.0, base * 0.25))))


def _sample_lead_time(params: SimulationParams, rng: np.random.Generator) -> float:
    mu = np.log(max(params.baseline_lead_time, 1.0))
    sigma = 0.25 + min(0.35, params.defect_rate_pct / 100.0)
    sampled = float(rng.lognormal(mu, sigma))
    sdi_factor = max(0.0, (params.supply_disruption_index - 4.09) / 5.88)
    inflation = (
        1.0
        + params.severity * 0.6
        + sdi_factor * 0.4
        + params.lead_time_inflation * 0.1
        + (0.25 if params.logistics_disruption else 0.0)
    )
    return max(1.0, sampled * inflation)


def _daily_demand(
    day: int,
    params: SimulationParams,
    disruption_end: int,
    rng: np.random.Generator,
) -> float:
    if params.forecast_daily_demands and day < len(params.forecast_daily_demands):
        base = params.forecast_daily_demands[day]
    else:
        base = params.mean_daily_demand

    noise = rng.normal(1.0, params.demand_cv)
    demand = max(0.0, base * noise)

    if day < disruption_end:
        shock = 1.0 + params.severity * 0.5 + params.composite_score * 0.3
        demand *= shock

    yield_loss = 1.0 - min(0.3, (params.defect_rate_pct / 100.0) * rng.uniform(0.8, 1.2))
    return max(0.0, demand * yield_loss)


# Monkey-patch helper avoided — use inline drop logic instead
def _apply_post_disruption_demand(
    day: int,
    disruption_end: int,
    demand: float,
    params: SimulationParams,
) -> float:
    if day >= disruption_end and params.forecast_daily_demands:
        return demand * (1.0 - min(0.5, params.composite_score * 0.25))
    return demand


def _run_single_trial(params: SimulationParams, rng: np.random.Generator) -> Tuple[bool, float, float, float]:
    lead_time = int(round(_sample_lead_time(params, rng)))
    disruption_duration = _sample_disruption_duration(params, rng)
    disruption_end = disruption_duration

    supplier_reliable = True
    if params.incoming_supply > 0:
        reliability = 0.55 + (1.0 - params.severity) * 0.35
        supplier_reliable = bool(rng.random() < reliability)

    inbound_qty = params.incoming_supply if supplier_reliable else 0.0
    inventory = params.initial_inventory
    inbound_schedule = {lead_time: inbound_qty}

    stockout = False
    first_stockout_day: Optional[float] = None
    unmet_demand = 0.0
    total_demand = 0.0
    peak_gap_pct = 0.0

    for day in range(params.horizon_days):
        if day in inbound_schedule:
            inventory += inbound_schedule[day]

        demand = _daily_demand(day, params, disruption_end, rng)
        demand = _apply_post_disruption_demand(day, disruption_end, demand, params)
        total_demand += demand

        if inventory >= demand:
            inventory -= demand
        else:
            shortfall = demand - inventory
            unmet_demand += shortfall
            if not stockout:
                stockout = True
                first_stockout_day = float(day + 1)
            inventory = 0.0

        if demand > 0:
            gap_pct = max(0.0, (1.0 - inventory / (demand + inventory + 1e-6)) * 100.0)
            peak_gap_pct = max(peak_gap_pct, gap_pct)

    revenue_loss = unmet_demand * params.unit_price_usd
    severity_score = min(100.0, (unmet_demand / max(total_demand, 1.0)) * 100.0)
    if stockout:
        severity_score = max(severity_score, 50.0)

    return stockout, first_stockout_day or float("nan"), revenue_loss, severity_score


def run_monte_carlo(params: SimulationParams) -> SimulationResult:
    rng = np.random.default_rng(params.seed)
    trials = max(100, params.trials)

    stockout_flags: List[bool] = []
    stockout_days: List[float] = []
    revenue_losses: List[float] = []
    severity_scores: List[float] = []

    for _ in range(trials):
        stockout, stockout_day, revenue_loss, severity_score = _run_single_trial(params, rng)
        stockout_flags.append(stockout)
        if stockout:
            stockout_days.append(stockout_day)
        revenue_losses.append(revenue_loss)
        severity_scores.append(severity_score)

    flags_arr = np.array(stockout_flags, dtype=float)
    severity_arr = np.array(severity_scores, dtype=float)
    revenue_arr = np.array(revenue_losses, dtype=float)
    stockout_day_arr = np.array(stockout_days, dtype=float) if stockout_days else np.array([])

    stockout_prob = float(flags_arr.mean() * 100.0)
    expected_gap = float(severity_arr.mean())

    sample_idx = np.linspace(0, len(revenue_arr) - 1, min(MAX_HISTOGRAM_SAMPLES, len(revenue_arr)), dtype=int)
    revenue_samples = [float(revenue_arr[i]) for i in sample_idx]

    return SimulationResult(
        stockout_probability_pct=round(stockout_prob, 2),
        expected_inventory_gap_pct=round(expected_gap, 2),
        alternate_route=params.alternate_route,
        stockout_probability_p10=round(_percentile(severity_arr, 10) or 0.0, 2),
        stockout_probability_p90=round(_percentile(severity_arr, 90) or 0.0, 2),
        days_to_stockout_p50=_percentile(stockout_day_arr, 50),
        days_to_stockout_p10=_percentile(stockout_day_arr, 10),
        days_to_stockout_p90=_percentile(stockout_day_arr, 90),
        revenue_impact_usd_p50=round(_percentile(revenue_arr, 50) or 0.0, 2),
        revenue_impact_usd_p10=round(_percentile(revenue_arr, 10) or 0.0, 2),
        revenue_impact_usd_p90=round(_percentile(revenue_arr, 90) or 0.0, 2),
        trials_run=trials,
        model_version=MODEL_VERSION,
        revenue_impact_samples=revenue_samples,
    )


def run_heuristic_fallback(params: SimulationParams) -> SimulationResult:
    """Last-resort deterministic estimate when Monte Carlo cannot run."""
    inventory = params.initial_inventory
    incoming = params.incoming_supply
    lead_time = params.baseline_lead_time
    stockout_probability = min(
        100.0,
        max(
            0.0,
            params.composite_score * 100.0
            + (1.0 - (inventory / (incoming + 1.0))) * 25.0
            + (lead_time / 30.0) * 25.0
            + params.severity * 15.0,
        ),
    )
    expected_gap = max(0.0, 100.0 - (inventory / (incoming + 1.0)) * 100.0)
    est_revenue = params.mean_daily_demand * params.unit_price_usd * (stockout_probability / 100.0) * 7
    return SimulationResult(
        stockout_probability_pct=round(stockout_probability, 2),
        expected_inventory_gap_pct=round(expected_gap, 2),
        alternate_route=params.alternate_route,
        stockout_probability_p10=round(stockout_probability * 0.6, 2),
        stockout_probability_p90=round(min(100.0, stockout_probability * 1.4), 2),
        revenue_impact_usd_p50=round(est_revenue, 2),
        revenue_impact_usd_p10=round(est_revenue * 0.5, 2),
        revenue_impact_usd_p90=round(est_revenue * 1.8, 2),
        trials_run=0,
        model_version="heuristic_fallback",
    )
