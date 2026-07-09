"""L6 — Monte Carlo simulation agent."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from src.agents.simulation_agent.engine import run_heuristic_fallback, run_monte_carlo
from src.agents.simulation_agent.priors import build_simulation_params
from src.agents.state import GlobalState
from src.utils.db_utils import insert_simulation_run

logger = logging.getLogger(__name__)

DEFAULT_TRIALS = 2000


def _trial_count() -> int:
    raw = os.environ.get("SIMULATION_TRIALS", str(DEFAULT_TRIALS))
    try:
        return max(100, int(raw))
    except ValueError:
        return DEFAULT_TRIALS


def simulation_agent(state: GlobalState) -> Dict[str, Any]:
    """L6 — Monte Carlo discrete-time inventory simulation."""
    if state.active_record is None or state.config is None:
        raise ValueError("Active record and config are required for simulation.")

    trials = _trial_count()
    seed = hash(
        (
            state.active_record.get("event_date"),
            state.active_record.get("port"),
            state.active_record.get("sku"),
            state.event_metadata.disruption_type if state.event_metadata else "",
        )
    ) % (2**31)

    try:
        params = build_simulation_params(state, trials=trials, seed=seed)
        result = run_monte_carlo(params)
        log_msg = f"L6: Monte Carlo simulation completed ({result.trials_run} trials)."
    except Exception as exc:
        logger.warning("L6 Monte Carlo failed, using heuristic fallback: %s", exc)
        params = build_simulation_params(state, trials=0, seed=seed)
        result = run_heuristic_fallback(params)
        log_msg = f"L6: Heuristic fallback used — {exc}"

    record = state.active_record
    meta = state.event_metadata
    payload = {
        "severity": meta.severity if meta else None,
        "shock_duration_days": meta.shock_duration_days if meta else None,
        "recovery_window_days": meta.recovery_window_days if meta else None,
        "disruption_type": meta.disruption_type if meta else None,
        "inventory_level": record.get("inventory_level"),
        "incoming_supply": record.get("incoming_supply"),
        "lead_time_days": record.get("lead_time_days"),
        "demand": record.get("demand"),
        "composite_score": state.risk_score_composite,
        "alternate_route": result.alternate_route,
        "model_version": result.model_version,
        "trials_run": result.trials_run,
    }

    try:
        insert_simulation_run(
            event_date=record.get("event_date") or record.get("order_date", ""),
            port=record.get("port", ""),
            sku=record.get("sku", ""),
            disruption_type=meta.disruption_type if meta else "",
            risk_label=state.risk_label or "",
            result=result,
            payload_json=json.dumps(payload),
        )
    except Exception as exc:
        logger.warning("L6: failed to persist simulation run: %s", exc)

    return {
        "simulation_result": result,
        "agent_logs": state.agent_logs + [log_msg],
    }
