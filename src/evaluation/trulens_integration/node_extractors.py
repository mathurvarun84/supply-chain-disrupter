"""
node_extractors.py — Pure functions that pull TruLens-relevant metrics out
of GlobalState after a node completes. No TruLens dependency; independently
testable.
"""

from __future__ import annotations

from typing import Optional

from src.agents.state import GlobalState

NODE_LATENCY_LABELS: dict[str, str] = {
    "l1_data_ingestion": "L1",
    "l2_news_analysis": "L2",
    "l3_weather_monitoring": "L3",
    "l4_risk_classifier": "L4",
    "l5_demand_forecast": "L5",
    "l6_simulation": "L6",
    "l7_mitigation": "L7",
}


def extract_l4_signals(state: GlobalState) -> Optional[dict]:
    """Pull the three L4 ensemble labels + composite score, or None if L4 hasn't run."""
    rc = state.risk_classification
    if rc is None:
        return None
    return {
        "composite_score": rc.composite_score,
        "rule_label": rc.rule_signal.escalated_label if rc.rule_signal else None,
        "distilbert_label": rc.distilbert_signal.predicted_label if rc.distilbert_signal else None,
        "llm_label": rc.llm_signal.predicted_label if rc.llm_signal else None,
    }


def extract_l5_forecast(state: GlobalState) -> Optional[dict]:
    """Pull the L5 forecast's expected demand drop, or None if L5 was skipped."""
    fr = state.forecast_result
    if fr is None:
        return None
    return {"expected_drop_pct": fr.expected_drop_pct}
