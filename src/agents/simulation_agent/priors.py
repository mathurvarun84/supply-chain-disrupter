"""Build simulation priors from GlobalState and config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.state import GlobalState
from src.utils.db_utils import fetch_ops_kpi_priors
from src.utils.yaml_utils import get_route_map

SEMICONDUCTOR_HUBS = {
    "Hsinchu": (24.80, 120.97),
    "Tainan": (22.99, 120.20),
    "Osaka": (34.69, 135.50),
    "Austin": (30.27, -97.74),
    "Shanghai": (31.23, 121.47),
    "Singapore": (1.35, 103.82),
    "Rotterdam": (51.92, 4.48),
    "Incheon": (37.46, 126.71),
    "Penang": (5.41, 100.33),
    "Ho_Chi_Minh_City": (10.82, 106.63),
    "Shenzhen": (22.54, 114.06),
    "Chennai": (13.08, 80.27),
}

LOGISTICS_DISRUPTION_TYPES = frozenset(
    {"port closure", "geopolitical", "supplier lockdown", "chip shortage"}
)

DEFAULT_BACKUP_ROUTE = "Cape of Good Hope"

REGION_ROUTE_DEFAULTS: Dict[str, str] = {
    "Eastern Asia": "Suez Canal",
    "Southeast Asia": "Suez Canal",
    "Western Europe": "Cape of Good Hope",
    "Northern Europe": "Cape of Good Hope",
    "North America": "Panama Canal",
    "South America": "Cape of Good Hope",
    "Middle East": "Suez Canal",
    "Africa": "Cape of Good Hope",
    "Oceania": "Suez Canal",
}


@dataclass
class SimulationParams:
    initial_inventory: float
    incoming_supply: float
    baseline_lead_time: float
    mean_daily_demand: float
    unit_price_usd: float
    horizon_days: int
    severity: float
    shock_duration_days: int
    disruption_type: str
    composite_score: float
    supply_disruption_index: float
    defect_rate_pct: float
    alternate_route: str
    logistics_disruption: bool
    forecast_daily_demands: List[float] = field(default_factory=list)
    expected_duration_days: Optional[float] = None
    lead_time_inflation: float = 0.0
    demand_cv: float = 0.15
    trials: int = 2000
    seed: Optional[int] = None


def _nearest_hub(lat: float, lon: float) -> str:
    best, best_dist = "Singapore", float("inf")
    for hub, (hlat, hlon) in SEMICONDUCTOR_HUBS.items():
        dist = math.sqrt((lat - hlat) ** 2 + (lon - hlon) ** 2)
        if dist < best_dist:
            best, best_dist = hub, dist
    return best


def resolve_alternate_route(config: Dict[str, Any], record: Dict[str, Any]) -> str:
    """Resolve backup route from port key, region map, or nearest hub."""
    port = record.get("port") or record.get("order_region") or ""
    route_maps = config.get("route_maps", {})
    if port in route_maps:
        route = route_maps[port].get("backup_route")
        if route:
            return route

    region_maps = config.get("region_route_maps", {})
    order_region = record.get("order_region") or port
    if order_region in region_maps:
        route = region_maps[order_region].get("backup_route")
        if route:
            return route
    if order_region in REGION_ROUTE_DEFAULTS:
        return REGION_ROUTE_DEFAULTS[order_region]

    lat = record.get("latitude")
    lon = record.get("longitude")
    if lat is not None and lon is not None:
        hub = _nearest_hub(float(lat), float(lon))
        hub_route = get_route_map(config, hub).get("backup_route")
        if hub_route:
            return hub_route
        if hub in REGION_ROUTE_DEFAULTS:
            return REGION_ROUTE_DEFAULTS[hub]

    legacy = get_route_map(config, port).get("backup_route")
    return legacy or DEFAULT_BACKUP_ROUTE


def _build_forecast_demands(state: GlobalState, horizon_days: int, fallback: float) -> List[float]:
    if not state.forecast_result or not state.forecast_result.prophet_forecast:
        return []
    demands: List[float] = []
    for point in state.forecast_result.prophet_forecast[:horizon_days]:
        yhat = float(point.get("yhat", fallback))
        demands.append(max(0.0, yhat))
    while len(demands) < horizon_days:
        demands.append(demands[-1] if demands else fallback)
    return demands[:horizon_days]


def build_simulation_params(state: GlobalState, trials: int, seed: Optional[int]) -> SimulationParams:
    if state.active_record is None or state.config is None:
        raise ValueError("Active record and config are required for simulation.")

    record = state.active_record
    meta = state.event_metadata

    inventory = float(record.get("inventory_level", 0.0))
    incoming = float(record.get("incoming_supply", 0.0))
    lead_time = max(float(record.get("lead_time_days", 7.0)), 1.0)
    demand = max(float(record.get("demand", 1.0)), 0.1)

    sales_usd = float(record.get("sales_usd", 0.0))
    unit_price = float(record.get("unit_price_usd", 0.0))
    if unit_price <= 0 and sales_usd > 0 and demand > 0:
        unit_price = sales_usd / demand
    if unit_price <= 0:
        unit_price = 1.0

    severity = float(meta.severity) if meta else 0.5
    shock_duration = int(meta.shock_duration_days) if meta else 0
    recovery_window = int(meta.recovery_window_days) if meta else 60
    disruption_type = (meta.disruption_type if meta else "unknown").lower()

    composite = float(state.risk_score_composite or 0.0)
    supply_idx = float(record.get("supply_disruption_index", 5.0))
    defect_rate = float(record.get("defect_rate_pct", 5.0))

    expected_duration: Optional[float] = None
    if state.news_analysis_llm and state.news_analysis_llm.expected_duration_days is not None:
        expected_duration = float(state.news_analysis_llm.expected_duration_days)

    sku = record.get("sku") or record.get("product_name") or ""
    region = record.get("order_region") or record.get("port") or ""
    ops_priors = fetch_ops_kpi_priors(sku, region)
    lead_time_inflation = 0.0
    demand_cv = 0.15
    if ops_priors:
        if ops_priors.get("mean_lead_time"):
            lead_time = max(lead_time, float(ops_priors["mean_lead_time"]))
        lead_time_inflation = float(ops_priors.get("mean_lead_time_inflation") or 0.0)
        if ops_priors.get("demand_cv"):
            demand_cv = float(ops_priors["demand_cv"])

    forecast_demands = _build_forecast_demands(state, recovery_window, demand)

    return SimulationParams(
        initial_inventory=inventory,
        incoming_supply=incoming,
        baseline_lead_time=lead_time,
        mean_daily_demand=demand,
        unit_price_usd=unit_price,
        horizon_days=recovery_window,
        severity=severity,
        shock_duration_days=shock_duration,
        disruption_type=disruption_type,
        composite_score=composite,
        supply_disruption_index=supply_idx,
        defect_rate_pct=defect_rate,
        alternate_route=resolve_alternate_route(state.config, record),
        logistics_disruption=disruption_type in LOGISTICS_DISRUPTION_TYPES,
        forecast_daily_demands=forecast_demands,
        expected_duration_days=expected_duration,
        lead_time_inflation=lead_time_inflation,
        demand_cv=demand_cv,
        trials=trials,
        seed=seed,
    )
