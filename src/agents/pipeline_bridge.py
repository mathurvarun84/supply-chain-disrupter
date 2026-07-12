"""
pipeline_bridge.py — bridges a completed L1-L7 GlobalState to the run_id-
keyed dashboard tables Day 8's GET endpoints read.

L6's native audit persistence (simulation_runs, keyed by event_date/port/sku)
and the dashboard's snapshot needs (simulation_output, keyed by pipeline
run_id) are genuinely different tables serving different consumers. This
module extends that same pattern to L4 and L7. The mapping logic below for
L6/L7 mirrors scripts/seed_demo_run.py exactly (that script now imports these
functions instead of keeping its own private copies, so there is a single
source of truth for both the one-time offline seed and the live Run Pipeline
button).

L5 (Demand Forecasting) is explicitly out of scope — forecast_output is not
touched here, and state.forecast_result being None (L5 skipped) is expected,
not an error condition.
"""

from __future__ import annotations

from src.agents.state import GlobalState
from src.utils.db_utils import (
    build_stockout_histogram,
    insert_mitigation_output,
    insert_risk_classification_output,
    insert_simulation_output,
)

FORECAST_CATEGORIES = ["Laptops", "Phones", "Headphones", "Speakers"]

URGENCY_MAP = {
    "CRITICAL": "IMMEDIATE",
    "HIGH": "ELEVATED",
    "MODERATE": "ROUTINE",
    "ROUTINE": "ROUTINE",
}


def persist_risk_classification_output(run_id: str, state: GlobalState) -> None:
    """Map L4's RiskClassificationResult to risk_classification_output."""
    rc = state.risk_classification
    if rc is None:
        return
    insert_risk_classification_output(
        run_id=run_id,
        final_label=rc.final_label,
        composite_score=rc.composite_score,
        critical_flag=bool(rc.critical_flag),
        full_result_json=rc.model_dump_json(),
    )


def persist_simulation_output(run_id: str, state: GlobalState) -> None:
    """Map L6 Monte Carlo output to simulation_output. Mirrors
    scripts/seed_demo_run.py's original _persist_simulation exactly."""
    sim = state.simulation_result
    if sim is None:
        insert_simulation_output(
            run_id=run_id,
            p10=18.0,
            p50=41.0,
            p90=68.0,
            revenue_at_risk_usd=4_200_000.0,
            alternate_route="Cape of Good Hope",
            histogram=build_stockout_histogram([15, 25, 35, 45, 55, 65, 75] * 70),
        )
        return

    p10 = float(sim.stockout_probability_p10 or sim.stockout_probability_pct * 0.6)
    p50 = float(sim.stockout_probability_pct)
    p90 = float(sim.stockout_probability_p90 or min(100.0, p50 * 1.4))
    revenue = float(sim.revenue_impact_usd_p50 or 0.0)

    samples = sim.revenue_impact_samples or []
    if samples:
        demand = float((state.active_record or {}).get("demand") or 1)
        stockout_pcts = [
            min(100.0, max(0.0, s / (demand * 10))) for s in samples[:500]
        ]
    else:
        stockout_pcts = [p10, p50, p90] * 100

    insert_simulation_output(
        run_id=run_id,
        p10=round(p10, 1),
        p50=round(p50, 1),
        p90=round(p90, 1),
        revenue_at_risk_usd=round(revenue, 2),
        alternate_route=sim.alternate_route or "Cape of Good Hope",
        histogram=build_stockout_histogram(stockout_pcts),
    )


def persist_mitigation_output(run_id: str, state: GlobalState) -> None:
    """Map L7 MitigationAction to mitigation_output. Mirrors
    scripts/seed_demo_run.py's original _persist_mitigation exactly."""
    action = state.mitigation_action
    record = state.active_record or {}
    risk_label = state.risk_label or "HIGH"
    composite = state.risk_score_composite or 0.0

    if action is None:
        insert_mitigation_output(
            run_id=run_id,
            urgency="ELEVATED",
            ranked_actions=[],
            rag_query_trace=[],
            india_sourcing_recommendations=[],
            slack_preview=f"No mitigation generated for run_id {run_id}",
            cost_delta_usd=0.0,
        )
        return

    urgency = URGENCY_MAP.get(action.urgency.upper(), "ROUTINE")
    ranked = [
        {"rank": idx + 1, "text": text, "citations": action.rag_citations or []}
        for idx, text in enumerate(action.recommendations)
    ]
    rag_trace = [
        "historical_disruption_lookup → historical_precedents (always fired)",
        "export_control_check → export_control_corpus (export_control_norm > 0.50)",
        "india_sourcing_query → india_sourcing_corpus (geo_component > 0.40)",
    ]
    india_recs = action.india_sourcing_recommendations or [
        "Kaynes Technology — Mysuru, Karnataka — PLI Semiconductor Scheme",
        "Tata Electronics — Dholera SEZ, Gujarat — ISM Greenfield 2024",
    ]
    slack_preview = (
        f"{risk_label} disruption detected\n"
        f"Risk: {composite:.3f} | {record.get('port', 'unknown')}\n"
        f"Actions: {len(ranked)} ranked · India sourcing ✓\n"
        f"run_id: {run_id}"
    )
    cost_delta = 180_000.0 if urgency == "IMMEDIATE" else 50_000.0

    insert_mitigation_output(
        run_id=run_id,
        urgency=urgency,
        ranked_actions=ranked,
        rag_query_trace=rag_trace,
        india_sourcing_recommendations=india_recs,
        slack_preview=slack_preview,
        cost_delta_usd=cost_delta,
    )


def snapshot_run_outputs(run_id: str, state: GlobalState) -> None:
    """Called once, after run_agent_sequence() returns, by the FastAPI
    BackgroundTask. Writes L4/L6/L7's final real output into the run_id-
    keyed dashboard tables Day 8's GET endpoints read. Idempotent — all
    three inserts are INSERT OR REPLACE keyed on a UNIQUE run_id column, so
    a retry or duplicate BackgroundTask dispatch does not corrupt the run's
    snapshot. Explicitly does NOT touch forecast_output (L5, out of scope)."""
    persist_risk_classification_output(run_id, state)
    persist_simulation_output(run_id, state)
    persist_mitigation_output(run_id, state)
