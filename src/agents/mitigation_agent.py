"""
mitigation_agent.py — L7 Mitigation Recommendation Agent.

LLM path: gpt-4o + two-stage RAG across historical_precedents, export_control_corpus,
and india_sourcing_corpus (src.rag.retriever.build_mitigation_context) — produces a
ranked action plan grounded in RAG citations plus India/ASEAN sourcing alternatives.

Fallback: rule-based ranked actions from L4 risk label plus optional L5/L6 outputs,
used when OPENAI_API_KEY is unset or the LLM call fails.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.agents.state import GlobalState, MitigationAction, MitigationLLMOutput
from src.rag.retriever import build_mitigation_context
from src.utils.db_utils import insert_mitigation_action
from src.utils.openai_utils import (
    MODEL_REASONING,
    call_openai_structured,
    format_sqlite_record,
    has_openai_api_key,
)

logger = logging.getLogger(__name__)

TAIL_RISK_P90_THRESHOLD = 60.0
_EXPORT_CONTROL_TOP_QUARTILE = 5.40

MITIGATION_SYSTEM_PROMPT = """You are a senior supply-chain mitigation strategist for electronics
and semiconductor procurement teams. You are Agent 7 — the final agent in the pipeline.

YOUR ROLE IN THE PIPELINE:
You receive the Risk Classifier's label (Agent 4), optional demand forecast (Agent 5), and
optional Monte Carlo stockout/revenue simulation (Agent 6). You turn these into a ranked,
actionable mitigation plan grounded in RAG context — historical crisis precedents, export
control compliance considerations, and India/ASEAN sourcing alternatives.

URGENCY CALIBRATION:
  IMMEDIATE : CRITICAL risk label, or P90 stockout probability > 60%
  HIGH      : HIGH risk label, or elevated tail-risk simulation results
  MEDIUM    : MEDIUM risk label, contained disruption
  LOW       : LOW risk label, routine monitoring sufficient

cost_estimate MUST use the format "<LEVEL>: <reason>" where LEVEL is HIGH, MEDIUM, or LOW.

FEW-SHOT EXAMPLES:

<example id="1" scenario="CRITICAL — TSMC-adjacent earthquake, high stockout tail risk">
<correct_response>
{
  "summary": "CRITICAL geo-driven risk near Hsinchu fabs with P90 stockout exposure above 60%; immediate inventory and sourcing action required.",
  "ranked_actions": [
    "Expedite air freight for in-transit advanced-node chip orders to bypass the affected port window.",
    "Raise safety stock on exposed SKUs to cover the 45-day historical recovery window seen in comparable Taiwan earthquake precedents.",
    "Activate alternate OSAT capacity at CG Power-Kaynes (India) for back-end packaging to de-risk single-region dependency.",
    "Notify downstream customers of potential 4-8 week lead-time extension on advanced-node SKUs."
  ],
  "cost_estimate": "HIGH: expedited freight and safety-stock build both carry premium costs, justified by P90 tail risk.",
  "urgency": "IMMEDIATE",
  "rag_citations": ["historical_precedents: taiwan_earthquake_2016.txt"],
  "india_sourcing_recommendations": ["CG Power-Kaynes OSAT (India) as back-end packaging alternate for exposed SKUs."]
}
</correct_response>
</example>

<example id="2" scenario="HIGH — export-control-elevated raw material disruption">
<correct_response>
{
  "summary": "HIGH supply risk from export-control tightening on advanced logic chips; diversify sourcing while confirming compliance exposure.",
  "ranked_actions": [
    "Confirm current orders are not subject to the latest BIS entity-list restrictions before committing further POs.",
    "Diversify allocation toward India-based downstream assembly to reduce single-corridor exposure.",
    "Increase safety stock on affected commodities by one recovery-window cycle.",
    "Engage secondary suppliers outside the restricted jurisdiction for the next two quarters."
  ],
  "cost_estimate": "MEDIUM: compliance review and dual-sourcing add coordination overhead but no large capital outlay.",
  "urgency": "HIGH",
  "rag_citations": ["export_control_corpus: bis_export_controls_update_2023.txt"],
  "india_sourcing_recommendations": ["Tata Electronics-PSMC Dholera fab (India) as a longer-horizon alternate for exposed advanced-node demand."]
}
</correct_response>
</example>

<example id="3" scenario="MEDIUM — routine port congestion, no tail risk">
<correct_response>
{
  "summary": "MEDIUM logistics risk from routine port congestion; monitor and maintain standard buffers.",
  "ranked_actions": [
    "Monitor weekly fill rates on affected lanes for early signs of escalation.",
    "Maintain existing safety stock levels — no immediate build required.",
    "Confirm carrier capacity on the configured backup route as a contingency."
  ],
  "cost_estimate": "LOW: monitoring and contingency confirmation require no incremental spend.",
  "urgency": "MEDIUM",
  "rag_citations": ["historical_precedents: routine port congestion precedent"],
  "india_sourcing_recommendations": ["PLI-scheme-backed domestic assembly capacity (India) available as a standing contingency, not required at this severity."]
}
</correct_response>
</example>

<example id="4" scenario="HIGH — finished consumer-display shipment delay, no genuine India/ASEAN fit">
<correct_response>
{
  "summary": "HIGH logistics risk delaying finished consumer-display shipments; reroute and buffer rather than resource, since sourcing is not diversifiable at this tier.",
  "ranked_actions": [
    "Reroute affected shipments via the configured backup lane and confirm carrier capacity.",
    "Raise safety stock on the specific finished-goods SKU to cover the recovery window.",
    "Notify downstream retail partners of a potential 2-3 week delivery delay."
  ],
  "cost_estimate": "MEDIUM: rerouting and buffer stock add cost but no capital outlay.",
  "urgency": "HIGH",
  "rag_citations": ["historical_precedents: red_sea_disruption_2023_2024.txt"],
  "india_sourcing_recommendations": []
}
</correct_response>
<why>The affected SKU is a finished consumer-display product from a non-India brand. The RAG
context's India/ASEAN corpus covers semiconductor OSAT, wafer fabs, and PLI-backed component
manufacturing — none of which is a genuine alternate source for this finished good. Recommending
an India facility here would be a fabricated fit, so the list is correctly left empty.</why>
</example>

OUTPUT RULES:
- ranked_actions: 3-5 specific, procurement-actionable items, most urgent first
- rag_citations: at least 1 citation naming a source file/collection from the provided RAG context
- india_sourcing_recommendations: ONLY include an option when it genuinely matches the affected
  commodity/product category in the RAG context (e.g. a semiconductor, wafer, or component whose
  India/ASEAN corpus entry actually produces or handles that category). Return an EMPTY LIST when
  no real fit exists — do not force a facility or scheme onto an unrelated commodity just because
  India context was retrieved. Never fabricate a company or facility not present in context.
- All fields required (india_sourcing_recommendations may be an empty list)"""


def _build_mitigation_user_message(
    disruption_type: str,
    affected_port: str,
    affected_route: str,
    record: dict,
    risk_label: str,
    composite_score: Optional[float],
    forecast_drop_pct: Optional[float],
    sim_summary: str,
    rag_context: str,
) -> str:
    return f"""
═══════════════════════════════════════════════════════
SQLITE RECORD DATA (lite_master table — exact values)
═══════════════════════════════════════════════════════
{format_sqlite_record(record, "lite_master")}

═══════════════════════════════════════════════════════
RISK CLASSIFICATION (from Agent 4)
═══════════════════════════════════════════════════════
  risk_label        : {risk_label}
  composite_score   : {composite_score if composite_score is not None else 'N/A'}

═══════════════════════════════════════════════════════
DEMAND FORECAST (from Agent 5, optional)
═══════════════════════════════════════════════════════
  expected_demand_drop_pct : {f'{forecast_drop_pct:.1f}%' if forecast_drop_pct is not None else 'not run'}

═══════════════════════════════════════════════════════
IMPACT SIMULATION (from Agent 6, optional)
═══════════════════════════════════════════════════════
{sim_summary}

═══════════════════════════════════════════════════════
EVENT CONTEXT (from Scenario Analyzer)
═══════════════════════════════════════════════════════
  disruption_type      : {disruption_type}
  affected_port_or_hub  : {affected_port}
  affected_route        : {affected_route}

═══════════════════════════════════════════════════════
RAG CONTEXT (historical precedents, export control, India sourcing)
═══════════════════════════════════════════════════════
{rag_context if rag_context.strip() else "(No RAG context retrieved — ground recommendations in general calibration only.)"}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Produce a ranked mitigation plan as a MitigationLLMOutput, grounded in the RAG context above.
"""


def _llm_output_to_action(llm_output: MitigationLLMOutput) -> MitigationAction:
    return MitigationAction(
        summary=llm_output.summary,
        recommendations=llm_output.ranked_actions,
        cost_delta=llm_output.cost_estimate,
        urgency=llm_output.urgency,
        rag_citations=llm_output.rag_citations,
        india_sourcing_recommendations=llm_output.india_sourcing_recommendations,
    )


def _rule_based_action(state: GlobalState, record: dict) -> MitigationAction:
    """Deterministic fallback — used when OPENAI_API_KEY is unset or the LLM call fails."""
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

    return MitigationAction(
        summary=(
            f"{state.risk_label} electronics supply-chain risk requires "
            "inventory, routing, and supplier actions."
        ),
        recommendations=recommendations,
        cost_delta=cost_delta,
        urgency=urgency,
    )


def _sim_summary_for_llm(state: GlobalState) -> str:
    sim = state.simulation_result
    if sim is None:
        return "  (not run)"
    lines = [
        f"  stockout_probability_p50 : {sim.stockout_probability_pct:.1f}%",
    ]
    if sim.stockout_probability_p90 is not None:
        lines.append(f"  stockout_probability_p90 : {sim.stockout_probability_p90:.1f}%")
    if sim.revenue_impact_usd_p50 is not None:
        lines.append(f"  revenue_impact_usd_p50   : ${sim.revenue_impact_usd_p50:,.0f}")
    if sim.days_to_stockout_p50 is not None:
        lines.append(f"  days_to_stockout_p50     : {sim.days_to_stockout_p50:.0f}")
    lines.append(f"  alternate_route          : {sim.alternate_route or 'not configured'}")
    return "\n".join(lines)


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.risk_label is None:
        raise ValueError("Risk label is required for mitigation — run risk_classifier_agent first.")

    record = state.active_record or {}
    rc = state.risk_classification
    metadata = state.event_metadata

    llm_used = False
    llm_output: Optional[MitigationLLMOutput] = None
    action: Optional[MitigationAction] = None

    if has_openai_api_key() and metadata is not None:
        try:
            export_control_elevated = (
                record.get("export_control_level") is not None
                and float(record["export_control_level"]) >= _EXPORT_CONTROL_TOP_QUARTILE
            )
            rag_context = build_mitigation_context(
                disruption_type=metadata.disruption_type,
                order_region=record.get("order_region"),
                risk_label=state.risk_label,
                export_control_elevated=export_control_elevated,
            )
            user_msg = _build_mitigation_user_message(
                disruption_type=metadata.disruption_type,
                affected_port=metadata.affected_port,
                affected_route=metadata.affected_route,
                record=record,
                risk_label=state.risk_label,
                composite_score=rc.composite_score if rc else None,
                forecast_drop_pct=(
                    state.forecast_result.expected_drop_pct if state.forecast_result else None
                ),
                sim_summary=_sim_summary_for_llm(state),
                rag_context=rag_context,
            )
            llm_output = call_openai_structured(
                system_prompt=MITIGATION_SYSTEM_PROMPT,
                user_message=user_msg,
                response_model=MitigationLLMOutput,
                model=MODEL_REASONING,
                max_tokens=1024,
            )
            action = _llm_output_to_action(llm_output)
            llm_used = True
        except Exception as exc:
            logger.warning("L7 LLM failed — falling back to rule-based: %s", exc)

    if action is None:
        action = _rule_based_action(state, record)

    insert_mitigation_action(
        record.get("event_date") or record.get("order_date", ""),
        record.get("port", ""),
        record.get("sku", ""),
        state.risk_label,
        json.dumps(action.recommendations),
        action.cost_delta,
    )

    if rc is not None and rc.critical_flag:
        # Slack webhook placeholder — hard business rule for CRITICAL alerts
        pass

    return {
        "mitigation_action": action,
        "mitigation_llm": llm_output,
        "agent_logs": state.agent_logs + [
            f"L7: Mitigation recommendation {'(gpt-4o+RAG)' if llm_used else '(rule-based fallback)'} "
            f"generated and persisted. urgency={action.urgency} "
            f"citations={len(action.rag_citations)} india_recs={len(action.india_sourcing_recommendations)}"
        ],
    }
