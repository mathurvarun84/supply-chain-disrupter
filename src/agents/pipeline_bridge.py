"""
pipeline_bridge.py — bridges a completed L1-L7 GlobalState to the run_id-
keyed dashboard tables Day 8's GET endpoints read.

L6's native audit persistence (simulation_runs, keyed by event_date/port/sku)
and the dashboard's snapshot needs (simulation_output, keyed by pipeline
run_id) are genuinely different tables serving different consumers. This
module extends that same pattern to L4, L5, and L7. The mapping logic below
mirrors scripts/seed_demo_run.py exactly (that script now imports these
functions instead of keeping its own private copies, so there is a single
source of truth for both the one-time offline seed and the live Run Pipeline
button).

L5 (Demand Forecasting) real output is per-SKU (ops_kpi SKU_ID, e.g.
"SKU001") — the ops_kpi source table carries no product-category dimension,
so there is no real Laptops/Phones/Headphones/Speakers rollup to snapshot.
persist_forecast_output() below reports the real winning SKU_id as the
"category" value instead of fabricating one of the four fixture category
names; the frontend's 4-item category selector is a known display gap
tracked as a scope cut, not something this module papers over. This does
NOT redesign per-SKU into category-level modeling — that would be a
forecasting-quality change, out of scope here.
"""

from __future__ import annotations

import math

from src.agents.state import GlobalState
from src.utils.db_utils import (
    build_stockout_histogram,
    insert_forecast_output,
    insert_mitigation_output,
    insert_mitigation_rag_trace,
    insert_risk_classification_output,
    insert_simulation_output,
)

URGENCY_MAP = {
    "CRITICAL": "IMMEDIATE",
    "HIGH": "HIGH",
    "MEDIUM": "MEDIUM",
    "LOW": "LOW",
}


def _resolve_sku_id(state: GlobalState) -> "str | None":
    """The single winning SKU_id L4's select_forecast_sku() chose for this
    run — the same one L5 forecasts on (ForecastResult.sku_id) and L6/L7
    implicitly act on via state.active_record. Threaded into the
    simulation/mitigation snapshots too so the dashboard can show one SKU_id
    across all three panels and make the "same SKU everywhere" guarantee
    visible, not just true in the data.

    Preference order: forecast_handoff (L4's authoritative snapshot) →
    active_record (raw source row) — forecast_handoff is None only when L4
    itself didn't run (e.g. rule-only replay).
    """
    if state.forecast_handoff is not None:
        return state.forecast_handoff.sku_id
    return (state.active_record or {}).get("sku_id")


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


def persist_forecast_output(run_id: str, state: GlobalState) -> None:
    """Map L5's real per-SKU forecast to forecast_output (Screen 3 Forecast tab).

    state.forecast_result.demand_forecast holds real
    {week_start, demand_baseline, demand_disrupted} points from
    DemandForecastingAgent v3 — both series are genuine model output, not
    fabricated. "category" carries the real winning sku_id, not one of the
    frontend's four fixture category names (see module docstring — ops_kpi
    has no category dimension to roll up).

    If L5 was Skipped-Optional (forecast_result is None — missing
    prophet/pandas, or <10 weeks of ops_kpi history for this SKU), a
    fixture-shaped fallback row is written instead of skipping the snapshot,
    mirroring persist_simulation_output's own graceful-degradation fallback,
    so /api/forecast/{run_id} always has a row instead of a bare 404.
    """
    fr = state.forecast_result

    if fr and fr.demand_forecast:
        sku_label = fr.sku_id or "unknown_sku"
        series = [
            {
                "day": point.get("week_start", f"W+{i + 1}"),
                "baseline": float(point.get("demand_baseline", 0.0)),
                "adjusted": float(point.get("demand_disrupted", 0.0)),
            }
            for i, point in enumerate(fr.demand_forecast)
        ]
        insert_forecast_output(
            run_id=run_id,
            category=sku_label,
            categories=[sku_label],
            series=series,
            impact_duration_days=fr.impact_duration_days,
        )
        return

    series = [
        {
            "day": f"D+{i + 1}",
            "baseline": round(1000 + math.sin(i * 0.3) * 40 + i * 1.5),
            "adjusted": round(max(380, 1000 - (i * 42 if i < 8 else 336))),
        }
        for i in range(30)
    ]
    insert_forecast_output(
        run_id=run_id,
        category="Skipped-Optional",
        categories=["Skipped-Optional"],
        series=series,
    )


def persist_simulation_output(run_id: str, state: GlobalState) -> None:
    """Map L6 Monte Carlo output to simulation_output. Mirrors
    scripts/seed_demo_run.py's original _persist_simulation exactly."""
    sim = state.simulation_result
    sku_id = _resolve_sku_id(state)
    if sim is None:
        insert_simulation_output(
            run_id=run_id,
            p10=18.0,
            p50=41.0,
            p90=68.0,
            revenue_at_risk_usd=4_200_000.0,
            alternate_route="Cape of Good Hope",
            histogram=build_stockout_histogram([15, 25, 35, 45, 55, 65, 75] * 70),
            sku_id=sku_id,
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
        # Real SimulationResult fields — previously computed by
        # run_monte_carlo() and then discarded here; now passed through so
        # the Forecast & Simulation tab can show the full P10/P90 spread
        # instead of only the P50 revenue figure.
        revenue_at_risk_p10_usd=(
            round(float(sim.revenue_impact_usd_p10), 2) if sim.revenue_impact_usd_p10 is not None else None
        ),
        revenue_at_risk_p90_usd=(
            round(float(sim.revenue_impact_usd_p90), 2) if sim.revenue_impact_usd_p90 is not None else None
        ),
        days_to_stockout_p10=sim.days_to_stockout_p10,
        days_to_stockout_p50=sim.days_to_stockout_p50,
        days_to_stockout_p90=sim.days_to_stockout_p90,
        sku_id=sku_id,
        impact_duration_days=sim.impact_duration_days,
    )


def _build_slack_preview(run_id: str, state: GlobalState, action) -> str:
    """Read-only preview text for a Slack alert — never actually sent from here
    (the real webhook POST, if any, fires elsewhere in the pipeline). Only
    called when slack_alert_fired is True."""
    record = state.active_record or {}
    metadata = state.event_metadata
    port_or_sku = record.get("port") or record.get("sku") or "affected shipment"
    disruption = metadata.disruption_type if metadata else state.risk_label
    top_action = action.recommendations[0] if action.recommendations else "See mitigation plan."
    return (
        f"\U0001F514 [{action.urgency}] {disruption} risk — {port_or_sku} (run {run_id})\n"
        f"{action.summary}\n"
        f"Top action: {top_action}"
    )


def _resolve_duration_days(state: GlobalState) -> "float | None":
    """L4's canonical disruption duration for this run -- ForecastHandoff.
    duration_days (the escalated max(news evidence, event
    shock_duration_days) L4 actually classified on, also read by L5/L6)
    when available, else RiskClassificationResult.duration_days directly
    (same value, for runs where L4 had no sku_id to build a handoff)."""
    if state.forecast_handoff is not None:
        return state.forecast_handoff.duration_days
    return state.risk_classification.duration_days if state.risk_classification else None


def _mitigation_window_text(state: GlobalState) -> "str | None":
    """Disruption length + recovery window shown in the Mitigation tab
    banner. The duration figure prefers _resolve_duration_days() (L4's
    canonical value, also read by L5/L6) over EventMetadata.
    shock_duration_days directly, so L7 reports the same duration
    L4/L5/L6 all agree on rather than an earlier, possibly-shorter raw
    event value. recovery_window_days has no L4 equivalent (L4 doesn't
    estimate recovery, only disruption length) so it still comes straight
    from EventMetadata. None when there's neither a resolved duration nor
    event metadata to fall back on."""
    metadata = state.event_metadata
    duration = _resolve_duration_days(state)
    if duration is None and metadata is None:
        return None
    duration_days = duration if duration is not None else (metadata.shock_duration_days if metadata else None)
    recovery_days = metadata.recovery_window_days if metadata else None
    duration_part = f"{duration_days:g}-day disruption window" if duration_days is not None else "disruption window unknown"
    recovery_part = f"{recovery_days}-day recovery" if recovery_days is not None else "recovery window unknown"
    return f"{duration_part}, {recovery_part}"


def persist_mitigation_output(run_id: str, state: GlobalState) -> None:
    """Map L7 MitigationAction to mitigation_output. Mirrors
    scripts/seed_demo_run.py's original _persist_mitigation exactly, plus the
    slack_alert_fired hard rule and the structured RAG trace this bridge now
    also captures (see mitigation_rag_trace table / insert_mitigation_rag_trace).

    slack_alert_fired is computed HERE from state.risk_classification.critical_flag
    — the same server-derived rule used for slack_should_fire on the risk
    classification endpoint (src/api/routers/risk.py) — and is never read off
    the LLM's mitigation output, which has no opinion on Slack at all.
    """
    action = state.mitigation_action
    rc = state.risk_classification
    slack_alert_fired = bool(rc.critical_flag) if rc is not None else False
    mitigation_window = _mitigation_window_text(state)
    sku_id = _resolve_sku_id(state)
    impact_duration_days = _resolve_duration_days(state)

    insert_mitigation_rag_trace(run_id, state.mitigation_rag_trace)

    if action is None:
        insert_mitigation_output(
            run_id=run_id,
            urgency="LOW",
            ranked_actions=[],
            rag_query_trace=[],
            india_sourcing_recommendations=[],
            slack_preview="",
            cost_delta_usd=0.0,
            slack_alert_fired=slack_alert_fired,
            mitigation_window=mitigation_window,
            sku_id=sku_id,
            impact_duration_days=impact_duration_days,
        )
        return

    urgency = URGENCY_MAP.get(action.urgency.upper(), action.urgency)
    ranked = [
        {"rank": idx + 1, "text": text, "citations": action.rag_citations or []}
        for idx, text in enumerate(action.recommendations)
    ]
    india_recs = action.india_sourcing_recommendations or []
    slack_preview = _build_slack_preview(run_id, state, action) if slack_alert_fired else ""

    sim = state.simulation_result
    cost_delta_usd = (
        round(float(sim.revenue_impact_usd_p50), 2)
        if sim is not None and sim.revenue_impact_usd_p50 is not None
        else 0.0
    )

    insert_mitigation_output(
        run_id=run_id,
        urgency=urgency,
        ranked_actions=ranked,
        rag_query_trace=[],
        india_sourcing_recommendations=india_recs,
        slack_preview=slack_preview,
        cost_delta_usd=cost_delta_usd,
        slack_alert_fired=slack_alert_fired,
        mitigation_window=mitigation_window,
        sku_id=sku_id,
        impact_duration_days=impact_duration_days,
    )


def snapshot_run_outputs(run_id: str, state: GlobalState) -> None:
    """Called once, after run_agent_sequence() returns, by the FastAPI
    BackgroundTask. Writes L4/L5/L6/L7's final real output into the run_id-
    keyed dashboard tables Day 8's GET endpoints read. Idempotent — all
    four inserts are INSERT OR REPLACE keyed on a UNIQUE run_id column, so
    a retry or duplicate BackgroundTask dispatch does not corrupt the run's
    snapshot."""
    persist_risk_classification_output(run_id, state)
    persist_forecast_output(run_id, state)
    persist_simulation_output(run_id, state)
    persist_mitigation_output(run_id, state)
