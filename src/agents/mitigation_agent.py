"""
mitigation_agent.py — L7 Mitigation Recommendation Agent.

Rule-based ranked actions from L4 risk label plus optional L5/L6 outputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.agents.state import GlobalState, MitigationAction
from src.utils.db_utils import insert_mitigation_action

logger = logging.getLogger(__name__)


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.risk_label is None:
        raise ValueError("Risk label is required for mitigation — run risk_classifier_agent first.")

    record = state.active_record or {}

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

    action = MitigationAction(
        summary=(
            f"{state.risk_label} electronics supply-chain risk requires "
            "inventory, routing, and supplier actions."
        ),
        recommendations=recommendations,
        cost_delta=cost_delta,
    )

    insert_mitigation_action(
        record.get("event_date") or record.get("order_date", ""),
        record.get("port", ""),
        record.get("sku", ""),
        state.risk_label,
        json.dumps(action.recommendations),
        action.cost_delta,
    )

    if state.risk_classification and state.risk_classification.critical_flag:
        # Slack webhook placeholder — hard business rule for CRITICAL alerts
        pass

    return {
        "mitigation_action": action,
        "agent_logs": state.agent_logs + ["L7: Mitigation recommendation generated and persisted."],
    }
