"""
Risk Classifier Agent — Agent 4 in the supply-chain disruption pipeline.

Responsibilities:
  - Normalize raw DB signals into [0, 1] component scores (geo, supply, freight, defect).
  - Derive a base risk label from delivery_status or composite score thresholds.
  - Apply the duration escalation matrix using news signals and event metadata.
  - Query ChromaDB for RAG grounding citations on HIGH/CRITICAL outcomes.
  - Persist the full audit record to risk_classifications and update lite_master (live mode only).

Two operating modes:
  REPLAY — stored composite + label exist in lite_master; trust the DB value, re-derive label
            from delivery_status for audit, never overwrite historical rows.
  LIVE   — new/injected order; recompute composite using the spec formula, write results back.
"""

import json
import logging
import sqlite3
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src.agents.distilbert_signal import run_distilbert_inference
from src.agents.risk_classifier_agent.judge_agent import run_judge
from src.agents.risk_classifier_agent.llm_signal import run_llm_signal
from src.agents.state import (
    ForecastHandoff,
    GlobalState,
    RiskClassificationResult,
    RuleBasedSignal,
)
from src.utils.db_utils import (
    ensure_risk_classification_table,
    execute_query,
    insert_risk_classification,
    update_risk_label,
)
from src.utils.etl_loader import SPEC_NORM_BOUNDS
from src.utils.guardrails import (
    log_guardrail_event,
    validate_input_null_fields,
    validate_output_hard_business_rule,
    validate_output_label_enum,
    validate_output_numeric_bounds,
)
from src.rag.utils import query_chroma_rag

logger = logging.getLogger(__name__)

# ── Normalization bounds — loaded once from SQLite at first call ──────────────

@lru_cache(maxsize=1)
def _get_norm_bounds() -> dict:
    """
    Read observed min/max from lite_master once and cache for the process lifetime.
    Bounds are logged on first load for auditability.
    """
    from src.utils.db_utils import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    try:
        q = conn.execute(
            """
            SELECT
                MIN(weather_severity_hub),   MAX(weather_severity_hub),
                MIN(natural_disaster_risk),  MAX(natural_disaster_risk),
                MIN(supply_disruption_index), MAX(supply_disruption_index),
                MIN(defect_rate_pct),         MAX(defect_rate_pct),
                MIN(disruption_news_count),   MAX(disruption_news_count)
            FROM lite_master
            """
        ).fetchone()
    finally:
        conn.close()

    bounds = {
        "weather_severity_hub":    (q[0] or 1.18,  q[1] or 10.0),
        "natural_disaster_risk":   (q[2] or 1.18,  q[3] or 10.0),
        "supply_disruption_index": (q[4] or 4.09,  q[5] or 9.97),
        "defect_rate_pct":         (q[6] or 2.0,   q[7] or 19.82),
        "disruption_news_count":   (q[8] or 0.0,   q[9] or SPEC_NORM_BOUNDS["disruption_news_count"][1]),
    }
    logger.info("Normalization bounds loaded from SQLite: %s", bounds)
    return bounds


def _norm(value: float, lo: float, hi: float) -> float:
    """Linear min-max normalize to [0, 1], clamped."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ── Component computation ─────────────────────────────────────────────────────

# Regional defect averages used when defect_rate_pct is NULL for an order.
_REGIONAL_DEFECT_AVGS: Dict[str, float] = {
    "Central Africa": 12.27, "Canada": 11.62, "West of USA": 11.18,
    "South America": 10.98, "Western Europe": 10.72, "West Asia": 10.68,
    "South of  USA": 10.61, "Eastern Asia": 10.40, "North Africa": 10.39,
    "US Center": 10.36,     "Southeast Asia": 10.33, "Southern Europe": 10.32,
    "Oceania": 10.32,       "Central America": 10.22, "South Asia": 10.21,
    "Central Asia": 10.07,  "East Africa": 10.07,  "East of USA": 9.93,
    "Southern Africa": 9.81, "Northern Europe": 9.74,
    "Caribbean": 9.63,      "West Africa": 9.63,   "Eastern Europe": 8.88,
}

# Composite score weights (must sum to 1.0)
_WEIGHTS = {"geo": 0.40, "supply": 0.30, "freight": 0.15, "defect": 0.15}


def _compute_components(
    live_weather_severity: Optional[float],
    natural_disaster_risk: Optional[float],
    supply_disruption_index: Optional[float],
    news_signals: list,
    defect_rate_pct: Optional[float],
    order_region: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute the four normalized [0, 1] component scores used by the live formula.

    geo_component    = max(norm(weather_severity_hub), norm(natural_disaster_risk))
    supply_component = norm(supply_disruption_index)
    freight_component= max news signal severity  (already 0-1, folds news into freight)
    defect_component = norm(defect_rate_pct)  — falls back to regional avg when None

    Note: live_weather_severity arrives pre-normalized from compute_weather_severity();
    natural_disaster_risk is on the raw DB scale and needs normalization.
    """
    bounds = _get_norm_bounds()

    w_norm = float(max(0.0, min(1.0, live_weather_severity))) if live_weather_severity is not None else 0.0
    d_raw = natural_disaster_risk if natural_disaster_risk is not None else 1.18
    geo = max(w_norm, _norm(d_raw, *bounds["natural_disaster_risk"]))

    s_raw = supply_disruption_index if supply_disruption_index is not None else 4.09
    supply = _norm(s_raw, *bounds["supply_disruption_index"])

    news_sev = max((sig.severity for sig in news_signals), default=0.0)
    freight = float(max(0.0, min(1.0, news_sev)))

    if defect_rate_pct is not None:
        d_val = defect_rate_pct
    else:
        d_val = _REGIONAL_DEFECT_AVGS.get(order_region or "", 10.36)
    defect = _norm(d_val, *bounds["defect_rate_pct"])

    return {"geo": geo, "supply": supply, "freight": freight, "defect": defect}


def _composite_from_components(components: Dict[str, float]) -> float:
    """Apply the weighted composite formula and round to 4 decimal places."""
    return round(
        _WEIGHTS["geo"] * components["geo"]
        + _WEIGHTS["supply"] * components["supply"]
        + _WEIGHTS["freight"] * components["freight"]
        + _WEIGHTS["defect"] * components["defect"],
        4,
    )


# ── Label derivation ──────────────────────────────────────────────────────────

def _base_label_from_delivery_status(
    delivery_status: Optional[str], composite_score: float
) -> str:
    """
    Map delivery_status (exact DataCo strings) to a base label.
    Falls back to composite_score thresholds when delivery_status is None/unrecognised.

    Exact delivery_status strings present in the dataset:
      "Shipping canceled"  → CRITICAL
      "Late delivery"      → HIGH
      "Advance shipping"   → LOW
      "Shipping on time"   → LOW
    Note: MEDIUM never comes from delivery_status; it only arises from the score fallback.
    """
    if delivery_status is not None:
        ds = delivery_status.strip()
        if ds == "Shipping canceled":
            return "CRITICAL"
        if ds == "Late delivery":
            return "HIGH"
        if ds in ("Advance shipping", "Shipping on time"):
            return "LOW"

    if composite_score >= 0.75:
        return "CRITICAL"
    if composite_score >= 0.50:
        return "HIGH"
    if composite_score >= 0.25:
        return "MEDIUM"
    return "LOW"


# ── Duration escalation ───────────────────────────────────────────────────────

_TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _max_duration_days(
    news_signals: list, event_metadata: Optional[Any]
) -> Optional[float]:
    """
    Return the maximum expected disruption duration (days) from:
      1. news_signals[i].expected_duration_days  — set by News Agent from RAG text
      2. event_metadata.shock_duration_days       — set from Scenario Analyzer UI
    Returns None when no duration signal exists.
    """
    candidates = []
    for sig in news_signals:
        if sig.expected_duration_days is not None:
            candidates.append(sig.expected_duration_days)
    if event_metadata is not None and hasattr(event_metadata, "shock_duration_days"):
        sdv = event_metadata.shock_duration_days
        if sdv is not None and sdv > 0:
            candidates.append(float(sdv))
    return max(candidates) if candidates else None


def _escalate_label(base_label: str, duration_days: Optional[float]) -> tuple:
    """
    Apply the duration escalation matrix. Returns (final_label, escalated).

    Matrix:
      duration_days is None or <= 1  → no change
      duration_days in [2, 3]        → escalate one tier (LOW→MEDIUM→HIGH→CRITICAL)
      duration_days >= 4             → force CRITICAL regardless of base label

    The classifier NEVER lowers a label; short duration does not de-escalate.
    """
    if duration_days is None or duration_days <= 1:
        return base_label, False

    if duration_days >= 4:
        final = "CRITICAL"
    else:
        idx = _TIER_ORDER.index(base_label) if base_label in _TIER_ORDER else 0
        final = _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]

    return final, final != base_label


def _apply_delivery_floor(label: str, delivery_override: Optional[str]) -> str:
    """
    Never allow ensemble fallback to produce a label below delivery_status override.

    "Shipping canceled" → CRITICAL and "Late delivery" → HIGH are hard floors.
    """
    if not delivery_override or delivery_override not in _TIER_ORDER:
        return label
    if label not in _TIER_ORDER:
        return delivery_override
    label_idx = _TIER_ORDER.index(label)
    floor_idx = _TIER_ORDER.index(delivery_override)
    return _TIER_ORDER[max(label_idx, floor_idx)]


# ── RAG grounding ─────────────────────────────────────────────────────────────

# Top-quartile threshold for export_control_level triggering an export-control RAG query.
# Derived as (max 6.629 - min 1.736) * 0.75 + 1.736 ≈ 5.40
_EXPORT_CONTROL_TOP_QUARTILE = 5.40


def _gather_rag_citations(
    final_label: str,
    escalated: bool,
    export_control_level: Optional[float] = None,
) -> tuple:
    """
    Query ChromaDB for grounding citations.

    RAG is triggered when:
      - final_label in ("HIGH", "CRITICAL"), or
      - escalated == True (label was just pushed up — most important grounding moment)
    Skipped for LOW/MEDIUM when not escalated (latency optimisation).

    Returns (citations: list[str], rationale: str).
    """
    if final_label not in ("HIGH", "CRITICAL") and not escalated:
        return [], "Low/medium risk — RAG grounding skipped."

    citations: List[str] = []
    rationale_parts: List[str] = []

    historical_hits = query_chroma_rag(
        "supply chain disruption historical precedent electronics semiconductor",
        n_results=3,
        where={"type": {"$in": ["static_report", "mitigation_playbook",
                                 "semiconductor_event", "event_profile"]}},
    )
    for hit in historical_hits:
        src = hit.get("metadata", {}).get("source", "")
        if src and src not in citations:
            citations.append(src)
        summary = hit.get("text", "")[:120].replace("\n", " ")
        if summary:
            rationale_parts.append(summary)

    if (
        export_control_level is not None
        and export_control_level >= _EXPORT_CONTROL_TOP_QUARTILE
    ):
        export_hits = query_chroma_rag(
            "export control sanctions BIS rule semiconductor restriction",
            n_results=2,
        )
        for hit in export_hits:
            if hit.get("distance", 1.0) > 0.6:
                continue
            src = hit.get("metadata", {}).get("source", "")
            if src and src not in citations:
                citations.append(src)
            summary = hit.get("text", "")[:100].replace("\n", " ")
            if summary:
                rationale_parts.append(f"[Export control] {summary}")

        if not any("export" in c.lower() or "BIS" in c for c in citations):
            rationale_parts.append("No export control grounding available in corpus.")

    rationale = " | ".join(rationale_parts) if rationale_parts else "No RAG context retrieved."
    return citations, rationale


# ── SQLite persistence helpers ────────────────────────────────────────────────

def _fetch_sdi_from_semiconductor_signals(year: Optional[Any]) -> Optional[float]:
    """
    Fetch the average supply_disruption_index for a given year from
    semiconductor_signals when the live-mode record does not carry one.
    """
    if year is None:
        return None
    rows = execute_query(
        "SELECT AVG(supply_disruption_index) FROM semiconductor_signals WHERE year = ?",
        (int(year),),
    )
    if rows and rows[0][0] is not None:
        return float(rows[0][0])
    return None


def _cross_check_known_severity(year: Optional[Any], final_label: str) -> None:
    """
    Compare computed label against semiconductor_signals.known_severity (log-only, non-blocking).
    """
    if year is None:
        return
    try:
        known_rows = execute_query(
            """
            SELECT known_severity FROM semiconductor_signals
            WHERE year = ? AND known_severity NOT IN ('LOW', '—', '-', 'None')
            LIMIT 1
            """,
            (int(year),),
        )
        if known_rows:
            known_sev = known_rows[0][0]
            if known_sev and known_sev != final_label:
                logger.warning(
                    "Label mismatch for year %s: computed=%s, known_severity=%s",
                    year, final_label, known_sev,
                )
    except Exception:
        pass  # cross-check is non-blocking


# ── Candidate selection (pre-L4-ensemble, cheap, no-LLM) ─────────────────────

_LABEL_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def select_forecast_sku(candidate_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pick exactly ONE record from N event-impacted candidates, BEFORE the
    expensive parts of the pipeline (L4's LLM ensemble, L5's Prophet fit)
    run on more than one. Forwarding multiple sku_ids to L5 means Prophet
    fits once per sku_id -- this function is what prevents that.

    Selection order (deterministic -- must never depend on dict/set
    iteration order or wall-clock time, or evaluation runs stop being
    reproducible):
      1. Highest disruption_event_label severity (CRITICAL > HIGH > MEDIUM > LOW)
      2. Highest risk_score_composite
      3. Lowest order_id (stable tie-break of last resort)

    REPLAY candidates already carry a stored risk_score_composite +
    disruption_event_label in lite_master -- use those directly, matching
    risk_classifier_agent()'s own REPLAY-mode trust-the-stored-value rule.
    Only a LIVE/injected candidate lacking a stored composite gets the
    cheap rule-based _compute_components()/_composite_from_components()
    scoring risk_classifier_agent() already uses for LIVE mode -- this
    function NEVER calls the LLM ensemble; that only runs once, afterward,
    on the single winner.
    """
    if not candidate_records:
        raise ValueError("select_forecast_sku() requires at least one candidate record.")
    if len(candidate_records) == 1:
        return candidate_records[0]

    def _rank_key(rec: Dict[str, Any]):
        label = rec.get("disruption_event_label")
        composite = rec.get("risk_score_composite")
        if label is None or composite is None or str(label).strip() in ("", "None", "nan"):
            # LIVE candidate with no stored value yet -- cheap rule-based
            # score only, never the LLM ensemble. Reuses the exact same
            # component/weight helpers risk_classifier_agent() uses for LIVE mode.
            components = _compute_components(
                live_weather_severity=None,   # no per-candidate live weather fetch here -- see SCOPE CUTS
                natural_disaster_risk=rec.get("natural_disaster_risk"),
                supply_disruption_index=rec.get("supply_disruption_index"),
                news_signals=[],
                defect_rate_pct=rec.get("defect_rate_pct"),
                order_region=rec.get("order_region"),
            )
            composite = _composite_from_components(components)
            label = _base_label_from_delivery_status(rec.get("delivery_status"), composite)
        order_id = rec.get("order_id") or rec.get("record_id") or 0
        return (_LABEL_RANK.get(label, 0), composite, -int(order_id))

    return max(candidate_records, key=_rank_key)


# ── Agent node ────────────────────────────────────────────────────────────────

def risk_classifier_agent(state: GlobalState) -> Dict[str, Any]:
    """
    Agent 4 — Risk Classifier.

    REPLAY mode: order already has risk_score_composite + disruption_event_label in
                 lite_master. Trust the stored composite; re-derive label from
                 delivery_status. Never overwrite historical SQLite rows.
    LIVE mode:   New/demo-injected order. Recompute composite from the spec formula
                 and write the result back to SQLite + risk_classifications.

    Duration escalation is applied in BOTH modes.
    """
    ensure_risk_classification_table()

    if state.event_metadata is None or state.active_record is None:
        raise ValueError("Data ingestion and record load are required for risk classification.")

    candidates = state.candidate_records or ([state.active_record] if state.active_record else [])
    if not candidates:
        raise ValueError("Data ingestion and record load are required for risk classification.")
    record = select_forecast_sku(candidates)
    if len(candidates) > 1:
        logger.info(
            "L4: %d candidates for this event; selected order_id=%s sku_id=%s (%s, composite=%s)",
            len(candidates), record.get("order_id"), record.get("sku_id"),
            record.get("disruption_event_label"), record.get("risk_score_composite"),
        )

    order_id = record.get("order_id") or record.get("record_id")
    sku_id = record.get("sku_id")  # None for rows loaded before the crosswalk work

    # ── Mode detection ────────────────────────────────────────────────────────
    stored_composite = record.get("risk_score_composite")
    stored_label = record.get("disruption_event_label")
    is_replay = (
        stored_composite is not None
        and stored_label is not None
        and str(stored_label).strip() not in ("", "None", "nan")
    )
    mode = "replay" if is_replay else "live"

    # ── Component computation ─────────────────────────────────────────────────
    # An active Scenario Analyzer simulation (a real disruption_type, not the
    # Screen 2 "historical_replay" sentinel set by _build_minimal_state in
    # src/api/routers/risk.py) must move the composite score, not just the
    # label. Every lite_master row already carries a precomputed composite
    # (ETL fills all rows), so mode == "replay" is true for nearly every
    # simulated order too -- freezing composite_score there made every
    # simulation against a historical order show the same dataset ground-truth
    # number regardless of the scenario's severity/duration.
    is_simulated_scenario = (
        state.event_metadata is not None
        and state.event_metadata.disruption_type != "historical_replay"
    )

    sdi = record.get("supply_disruption_index")
    if mode == "live" and sdi is None:
        sdi = _fetch_sdi_from_semiconductor_signals(record.get("year"))

    # Guardrail (input) — null/missing critical-field gate before the composite calc.
    # _compute_components() already has deterministic fallbacks for these fields
    # (regional defect average, bounds-default raw values) — this only detects/logs.
    null_check = validate_input_null_fields(
        {"supply_disruption_index": sdi, "export_control_level": record.get("export_control_level")},
        ["supply_disruption_index", "export_control_level"],
    )
    log_guardrail_event(
        agent_name="L4_risk_classifier", guardrail_name=null_check.guardrail_name, direction="input",
        passed=null_check.passed, reason=null_check.reason, record_id=state.run_id,
    )

    components = _compute_components(
        live_weather_severity=state.live_weather_severity,
        natural_disaster_risk=record.get("natural_disaster_risk"),
        supply_disruption_index=sdi,
        news_signals=state.news_signals,
        defect_rate_pct=record.get("defect_rate_pct"),
        order_region=record.get("order_region"),
    )

    if mode == "replay" and not is_simulated_scenario:
        composite_score = float(stored_composite)
    else:
        composite_score = _composite_from_components(components)

    # Guardrail (output) — composite_score must lie in [0, 1].
    bounds_check = validate_output_numeric_bounds(composite_score, "composite_score")
    log_guardrail_event(
        agent_name="L4_risk_classifier", guardrail_name=bounds_check.guardrail_name, direction="output",
        passed=bounds_check.passed, reason=bounds_check.reason, record_id=state.run_id,
    )

    # ── Label derivation ──────────────────────────────────────────────────────
    base_label = _base_label_from_delivery_status(record.get("delivery_status"), composite_score)

    # ── Duration escalation (duration also fed to DistilBERT as input) ─────────
    duration_days = _max_duration_days(state.news_signals, state.event_metadata)
    final_label, escalated = _escalate_label(base_label, duration_days)

    # ── RAG grounding ─────────────────────────────────────────────────────────
    rag_citations, rationale = _gather_rag_citations(
        final_label, escalated, record.get("export_control_level")
    )

    # ── Cross-check against known_severity (audit log only) ──────────────────
    _cross_check_known_severity(record.get("year"), final_label)

    # ── Delivery status override tracking ────────────────────────────────────
    delivery_override = None
    ds = record.get("delivery_status")
    if ds is not None:
        ds_stripped = str(ds).strip()
        if ds_stripped == "Shipping canceled":
            delivery_override = "CRITICAL"
        elif ds_stripped == "Late delivery":
            delivery_override = "HIGH"

    # ── Signal 1: package rule-based computation ─────────────────────────────
    rule_signal = RuleBasedSignal(
        composite_score=composite_score,
        geo_component=components["geo"],
        supply_component=components["supply"],
        freight_component=components["freight"],
        defect_component=components["defect"],
        base_label=base_label,
        escalated_label=final_label,
        escalated=escalated,
        duration_days=duration_days,
        delivery_status_override=delivery_override,
    )

    # ── Signal 2: DistilBERT (non-blocking) ──────────────────────────────────
    distilbert_signal = run_distilbert_inference(record, duration_days=duration_days)

    # Guardrail (output) — DistilBERTSignal.confidence has no Pydantic bound today.
    confidence_check = validate_output_numeric_bounds(distilbert_signal.confidence, "distilbert_signal.confidence")
    log_guardrail_event(
        agent_name="L4_risk_classifier", guardrail_name=confidence_check.guardrail_name, direction="output",
        passed=confidence_check.passed, reason=confidence_check.reason, record_id=state.run_id,
    )

    # ── Fetch semiconductor signals for Signal 3 + Judge ─────────────────────
    semiconductor_rows: List[dict] = []
    if record.get("year"):
        try:
            rows = execute_query(
                "SELECT year, company, supply_disruption_index, export_control_level, "
                "known_disruption_event, known_severity FROM semiconductor_signals "
                "WHERE year = ? ORDER BY supply_disruption_index DESC LIMIT 5",
                (int(record["year"]),),
            )
            semiconductor_rows = [dict(r) for r in rows]
        except Exception:
            pass

    # ── Signal 3: GPT-4o + two-stage RAG (non-blocking) ──────────────────────
    llm_signal = None
    if state.event_metadata is not None:
        llm_signal = run_llm_signal(
            record=record,
            semiconductor_rows=semiconductor_rows,
            rule_signal=rule_signal,
            disruption_type=state.event_metadata.disruption_type,
            order_region=record.get("order_region"),
            run_id=state.run_id,
            trace=state.langfuse_trace,
            span=state.langfuse_span,
        )

    # ── LLM-as-Judge (non-blocking) ──────────────────────────────────────────
    judge_verdict = run_judge(
        rule_signal=rule_signal,
        distilbert_signal=distilbert_signal,
        llm_signal=llm_signal,
        record=record,
        semiconductor_rows=semiconductor_rows,
        run_id=state.run_id,
        trace=state.langfuse_trace,
        span=state.langfuse_span,
    )

    # ── Final label fallback chain: judge → llm_signal → rule-based ──────────
    if judge_verdict is not None:
        final_label = judge_verdict.final_label
    elif llm_signal is not None:
        final_label = llm_signal.predicted_label
    else:
        final_label = rule_signal.escalated_label

    final_label = _apply_delivery_floor(final_label, rule_signal.delivery_status_override)

    # critical_flag derived from final_label — never from judge alone
    critical_flag = (final_label == "CRITICAL")

    # Guardrail (output) — label-enum enforcement. RiskClassificationResult.final_label
    # is a plain str, not a Pydantic Literal, so this is real detection, not just logging.
    label_check = validate_output_label_enum(final_label)
    log_guardrail_event(
        agent_name="L4_risk_classifier", guardrail_name=label_check.guardrail_name, direction="output",
        passed=label_check.passed, reason=label_check.reason, record_id=state.run_id,
    )

    # Guardrail (output) — hard business-rule override. Confirms critical_flag agrees
    # with final_label; this WRAPS the enforcement above (delivery-status floor, judge's
    # Shipping-canceled force-CRITICAL) with a logged event, per constraint C.
    hard_rule_check = validate_output_hard_business_rule(final_label, critical_flag)
    log_guardrail_event(
        agent_name="L4_risk_classifier", guardrail_name=hard_rule_check.guardrail_name, direction="output",
        passed=hard_rule_check.passed, reason=hard_rule_check.reason, record_id=state.run_id,
    )

    # LLM display fields from Signal 3 or judge reasoning
    llm_enhanced_rationale = None
    llm_evaluator_one_liner = None
    llm_primary_driver = None
    llm_confidence = None
    if llm_signal is not None:
        llm_enhanced_rationale = llm_signal.rationale
        llm_primary_driver = llm_signal.primary_driver
        llm_confidence = llm_signal.confidence_level
        llm_evaluator_one_liner = (
            f"{llm_signal.predicted_label} — {llm_signal.primary_driver} driver "
            f"({llm_signal.confidence_level} confidence)"
        )[:80]
    if judge_verdict is not None and judge_verdict.disagreement_explanation:
        llm_enhanced_rationale = (
            (llm_enhanced_rationale or "") + "\n\n[Judge] " + judge_verdict.reasoning
        )

    # ── Ensemble summary log ─────────────────────────────────────────────────
    db_label = distilbert_signal.predicted_label
    llm_label = llm_signal.predicted_label if llm_signal else "N/A"
    jv_label = judge_verdict.final_label if judge_verdict else "N/A"
    ensemble_summary = (
        f"ENSEMBLE | rule={rule_signal.escalated_label} | "
        f"distilbert={db_label}({distilbert_signal.confidence:.0%} conf) | "
        f"llm={llm_label} | judge={jv_label}"
    )
    logger.info("L4: %s", ensemble_summary)

    # ── Persist classification audit record ───────────────────────────────────
    # full_result_json carries the complete ensemble (rule + distilbert + llm +
    # judge) so a later cache-hit read for this order_id can still render the
    # full 3-signal ensemble, not just the rule-based column — see
    # src/api/routers/risk.py::_response_from_cached_row.
    insert_risk_classification(
        order_id=order_id,
        mode=mode,
        composite_score=composite_score,
        geo_component=components["geo"],
        supply_component=components["supply"],
        freight_component=components["freight"],
        defect_component=components["defect"],
        duration_days=duration_days,
        base_label=base_label,
        final_label=final_label,
        escalated=escalated,
        rag_citations=rag_citations,
        rationale=rationale,
        sku_id=sku_id,
        full_result_json=json.dumps(
            {
                "rule_signal": rule_signal.model_dump(),
                "distilbert_signal": distilbert_signal.model_dump() if distilbert_signal else None,
                "llm_signal": llm_signal.model_dump() if llm_signal else None,
                "judge_verdict": judge_verdict.model_dump() if judge_verdict else None,
            }
        ),
    )

    # Live mode: propagate freshly computed score/label back to lite_master.
    # REPLAY rows are never overwritten (that would corrupt Day-23 ground truth).
    if mode == "live":
        event_date = record.get("event_date") or record.get("order_date", "")
        update_risk_label(
            event_date,
            record.get("port", ""),
            record.get("sku", ""),
            composite_score,
            final_label,
        )

    result = RiskClassificationResult(
        mode=mode,
        composite_score=composite_score,
        geo_component=components["geo"],
        supply_component=components["supply"],
        freight_component=components["freight"],
        defect_component=components["defect"],
        duration_days=duration_days,
        base_label=base_label,
        final_label=final_label,
        escalated=escalated,
        rag_citations=rag_citations,
        rationale=rationale,
        critical_flag=critical_flag,
        llm_enhanced_rationale=llm_enhanced_rationale,
        llm_evaluator_one_liner=llm_evaluator_one_liner,
        llm_primary_driver=llm_primary_driver,
        llm_confidence=llm_confidence,
        rule_signal=rule_signal,
        distilbert_signal=distilbert_signal,
        llm_signal=llm_signal,
        judge_verdict=judge_verdict,
        sku_id=sku_id,
    )

    # ── L4 -> L5 handoff ──────────────────────────────────────────────────────
    # Only ever built from the winning record `select_forecast_sku()` chose --
    # never fabricated when sku_id is missing (pre-crosswalk data).
    forecast_handoff = None
    if sku_id:
        forecast_handoff = ForecastHandoff(
            sku_id=sku_id,
            order_id=record.get("order_id"),
            product_name=record.get("product_name"),
            category_name=record.get("category_name"),
            order_date=record.get("order_date"),
            unit_price_usd=record.get("unit_price_usd"),
            sales_usd=record.get("sales_usd"),
            risk_score_composite=composite_score,
            risk_label=final_label,
            candidates_considered=len(candidates),
            duration_days=duration_days,
        )

    llm_tag = "(ensemble+gpt-4o)" if llm_signal else "(ensemble rule-only)"
    return {
        "risk_classification": result,
        "judge_verdict": judge_verdict,
        "forecast_handoff": forecast_handoff,
        "agent_logs": state.agent_logs + [
            f"L4: Risk classification {llm_tag}. mode={mode} "
            f"composite={composite_score:.3f} base={base_label} "
            f"final={final_label} escalated={escalated} "
            f"duration={duration_days}d citations={len(rag_citations)} | {ensemble_summary}"
        ],
    }
