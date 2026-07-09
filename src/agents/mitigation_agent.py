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

TAIL_RISK_P90_THRESHOLD = 60.0


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.risk_label is None:
        raise ValueError("Risk label is required for mitigation — run risk_classifier_agent first.")

    record = state.active_record or {}
    sim = state.simulation_result

    stockout_p50 = sim.stockout_probability_pct if sim else None
    stockout_p90 = sim.stockout_probability_p90 if sim else None
    revenue_p50 = sim.revenue_impact_usd_p50 if sim else None
    forecast_drop = state.forecast_result.expected_drop_pct if state.forecast_result else None
    alt_route = (
        sim.alternate_route if sim else "the configured backup route"
    ) or "the configured backup route"

    if stockout_p50 is not None and stockout_p90 is not None:
        stockout_note = f"P50 {stockout_p50:.1f}% / P90 {stockout_p90:.1f}%"
        stockout_rec = (
            f"Raise safety stock — stockout severity range: {stockout_note}. "
            f"{'Prioritize buffer inventory: P90 exceeds 40%.' if stockout_p90 > 40 else 'Monitor weekly fill rates.'}"
        )
    else:
        stockout_note = "unknown (simulation not run)"
        stockout_rec = f"Raise safety stock for the affected product — stockout estimate: {stockout_note}."

    if revenue_p50 is not None:
        revenue_note = f"${revenue_p50:,.0f}"
    else:
        revenue_note = "unknown (simulation not run)"

    forecast_note = f"{forecast_drop:.1f}%" if forecast_drop is not None else "unknown (forecast not run)"

    recommendations = [
        stockout_rec,
        f"Prepare diversion through {alt_route} and confirm carrier capacity.",
        f"Review alternate suppliers — forecast variance: {forecast_note}; estimated revenue at risk (P50): {revenue_note}.",
    ]

    urgency = "HIGH"
    if state.risk_label == "CRITICAL":
        urgency = "CRITICAL"
    elif stockout_p90 is not None and stockout_p90 > TAIL_RISK_P90_THRESHOLD:
        urgency = "CRITICAL"
    elif state.risk_label in ("HIGH", "CRITICAL"):
        urgency = "HIGH"
    else:
        urgency = "MODERATE"

    cost_delta = (
        "High: expedite critical inventory and activate alternate sourcing."
        if urgency == "CRITICAL"
        else "Moderate: reserve backup logistics and inventory capacity."
    )

    action = MitigationAction(
        summary=(
            f"{state.risk_label} electronics supply-chain risk requires "
            "inventory, routing, and supplier actions."
        ),
        recommendations=recommendations,
        cost_delta=cost_delta,
        urgency=urgency,
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
