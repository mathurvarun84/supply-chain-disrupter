"""
Risk Classification (Screen 2) endpoints — L4 ensemble reader.

GET /api/risk-classification/{run_id} — run_id is the order_id (int,
stringified). Reads the persisted risk_classifications row for that
order if one exists (cheap — no re-running the ensemble, no repeat
OpenAI calls); the persisted row only carries the rule-signal columns
(see ensure_risk_classification_table), so a cache hit returns
distilbert_signal/llm_signal/judge_verdict as unavailable and
from_cache=True, documented honestly rather than faked. On a cache
miss, resolves the order to a daily_records row, builds a minimal
GlobalState, and calls risk_classifier_agent() directly — same
"call the agent directly, don't run the full graph" pattern used for
Day 2's Live Feed screen. The agent's own insert_risk_classification
call persists the result, so this router never writes to SQLite itself.

GET /api/risk-classification/latest — resolves to the most recently
classified (or most recent) order_id and delegates to the same logic,
for the frontend's default view before a demo-scenario/record picker
exists (Day 9's job). Registered before /{run_id} so FastAPI doesn't
swallow "latest" as a run_id.

Used by: TabRiskClassification.tsx (Screen 2).
"""

from typing import Optional

import json
import logging

from fastapi import APIRouter, HTTPException

from src.agents.risk_classifier_agent.agent import risk_classifier_agent
from src.agents.state import EventMetadata, GlobalState, RiskClassificationResult
from src.api.risk_classification_schemas import (
    DistilBertSignalResponse,
    JudgeVerdictResponse,
    LlmSignalResponse,
    RiskClassificationResponse,
    RuleSignalResponse,
)
from src.utils.db_utils import (
    fetch_latest_classified_order_id,
    fetch_record_by_order_id,
    fetch_risk_classification,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_minimal_state(record: dict) -> GlobalState:
    """Construct the smallest GlobalState risk_classifier_agent needs to
    run against a stored historical record, outside the live LangGraph
    pipeline. REPLAY-mode records (which is what daily_records holds)
    already carry their own stored composite/label and ignore
    event_metadata's duration fields, so this is a thin placeholder, not
    a general-purpose state builder — do not extend beyond this
    endpoint's needs."""
    event_metadata = EventMetadata(
        disruption_type="historical_replay",
        affected_port=record.get("port") or "",
        affected_route="",
        severity=0.0,
        shock_duration_days=0,
        recovery_window_days=0,
        synthetic_ratio=0.0,
    )
    return GlobalState(event_metadata=event_metadata, active_record=record)


def _response_from_cached_row(order_id: int, row: dict) -> RiskClassificationResponse:
    """Build a response from a persisted risk_classifications row.

    Rows written after the full_result_json column was added (see
    insert_risk_classification) carry the complete serialized ensemble
    (rule + distilbert + llm + judge), so a cache hit can return the full
    3-signal detail exactly as the original live-compute call saw it.
    Legacy rows written before that column existed have it NULL — for
    those, DistilBERT/LLM/Judge are reported as unavailable rather than
    fabricated, and from_cache=True lets the frontend show that honestly.
    """
    final_label = row["final_label"]
    critical_flag = final_label == "CRITICAL"
    rule_signal = RuleSignalResponse(
        composite_score=row["composite_score"],
        geo_component=row["geo_component"],
        supply_component=row["supply_component"],
        freight_component=row["freight_component"],
        defect_component=row["defect_component"],
        base_label=row["base_label"],
        escalated_label=final_label,
        escalated=bool(row["escalated"]),
        duration_days=row["duration_days"],
        delivery_status_override=None,
    )

    full_result_json = row["full_result_json"] if "full_result_json" in row.keys() else None
    if not full_result_json:
        return RiskClassificationResponse(
            run_id=str(order_id),
            order_id=order_id,
            mode=row["mode"],
            rule_signal=rule_signal,
            distilbert_signal=DistilBertSignalResponse(),
            llm_signal=LlmSignalResponse(),
            judge_verdict=None,
            final_label=final_label,
            final_critical_flag=critical_flag,
            slack_should_fire=critical_flag,
            from_cache=True,
        )

    parsed = json.loads(full_result_json)
    db_sig = parsed.get("distilbert_signal")
    llm_sig = parsed.get("llm_signal")
    jv = parsed.get("judge_verdict")
    return RiskClassificationResponse(
        run_id=str(order_id),
        order_id=order_id,
        mode=row["mode"],
        rule_signal=rule_signal,
        distilbert_signal=DistilBertSignalResponse(**db_sig) if db_sig else DistilBertSignalResponse(),
        llm_signal=LlmSignalResponse(**llm_sig) if llm_sig else LlmSignalResponse(),
        judge_verdict=JudgeVerdictResponse(**jv) if jv else None,
        final_label=final_label,
        final_critical_flag=critical_flag,
        slack_should_fire=critical_flag,
        from_cache=True,
    )


def _response_from_result(
    order_id: int, result: RiskClassificationResult
) -> RiskClassificationResponse:
    """Build a response from a freshly computed RiskClassificationResult
    (the live-compute path). final_critical_flag/slack_should_fire are
    recomputed here from final_label — never passed through from
    result.critical_flag directly — matching the escalation-guard
    contract (never trust an LLM-derived flag)."""
    rs = result.rule_signal
    rule_signal = RuleSignalResponse(
        composite_score=rs.composite_score,
        geo_component=rs.geo_component,
        supply_component=rs.supply_component,
        freight_component=rs.freight_component,
        defect_component=rs.defect_component,
        base_label=rs.base_label,
        escalated_label=rs.escalated_label,
        escalated=rs.escalated,
        duration_days=rs.duration_days,
        delivery_status_override=rs.delivery_status_override,
    )

    db_sig = result.distilbert_signal
    distilbert_signal = DistilBertSignalResponse(
        predicted_label=db_sig.predicted_label if db_sig else None,
        confidence=db_sig.confidence if db_sig else None,
        probability_distribution=db_sig.probability_distribution if db_sig else {},
        model_source=db_sig.model_source if db_sig else "not-available-skipped",
        inference_ms=db_sig.inference_ms if db_sig else None,
    )

    llm_sig = result.llm_signal
    llm_signal = LlmSignalResponse(
        predicted_label=llm_sig.predicted_label if llm_sig else None,
        rationale=llm_sig.rationale if llm_sig else None,
        rag_citations=llm_sig.rag_citations if llm_sig else [],
        rag_chunks_used=llm_sig.rag_chunks_used if llm_sig else 0,
        confidence_level=llm_sig.confidence_level if llm_sig else None,
        primary_driver=llm_sig.primary_driver if llm_sig else None,
    )

    jv = result.judge_verdict
    judge_verdict = (
        JudgeVerdictResponse(
            final_label=jv.final_label,
            verdict_type=jv.verdict_type,
            reasoning=jv.reasoning,
            signals_agreed=jv.signals_agreed,
            disagreement_explanation=jv.disagreement_explanation,
        )
        if jv is not None
        else None
    )

    final_label = result.final_label
    critical_flag = final_label == "CRITICAL"
    return RiskClassificationResponse(
        run_id=str(order_id),
        order_id=order_id,
        mode=result.mode,
        rule_signal=rule_signal,
        distilbert_signal=distilbert_signal,
        llm_signal=llm_signal,
        judge_verdict=judge_verdict,
        final_label=final_label,
        final_critical_flag=critical_flag,
        slack_should_fire=critical_flag,
        from_cache=False,
    )


def _classify_order(order_id: int) -> RiskClassificationResponse:
    cached = fetch_risk_classification(order_id)
    if cached is not None:
        return _response_from_cached_row(order_id, cached)

    record = fetch_record_by_order_id(order_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record found for order_id={order_id}")

    state = _build_minimal_state(record)
    try:
        output = risk_classifier_agent(state)
    except (SystemExit, KeyboardInterrupt, GeneratorExit):
        raise
    except BaseException as exc:
        # risk_classifier_agent's own signals (DistilBERT/LLM/Judge) are
        # documented as non-blocking, but its RAG citation lookup
        # (_gather_rag_citations -> ChromaDB) has no such guard today and
        # can raise on an environment-level ChromaDB failure (a
        # chromadb package/on-disk-store version mismatch observed
        # locally). That failure surfaces as pyo3_runtime.PanicException,
        # which subclasses BaseException (not Exception) — a plain
        # `except Exception` does NOT catch it, hence the broader clause
        # here with control-flow exceptions explicitly re-raised above.
        # Surfaced as a clear 503 instead of a raw crash; the underlying
        # ChromaDB issue still needs fixing separately.
        logger.exception("risk_classifier_agent failed for order_id=%s", order_id)
        raise HTTPException(
            status_code=503,
            detail=f"Ensemble computation failed for order_id={order_id}: {exc}",
        )
    result: RiskClassificationResult = output["risk_classification"]
    return _response_from_result(order_id, result)


@router.get("/latest", response_model=RiskClassificationResponse)
def get_latest_risk_classification() -> RiskClassificationResponse:
    """Return the ensemble result for the most recently classified (or
    most recent) order — Screen 2's default view until a demo-scenario/
    record picker exists (Day 9)."""
    order_id: Optional[int] = fetch_latest_classified_order_id()
    if order_id is None:
        raise HTTPException(status_code=404, detail="No orders available to classify.")
    return _classify_order(order_id)


@router.get("/{run_id}", response_model=RiskClassificationResponse)
def get_risk_classification(run_id: str) -> RiskClassificationResponse:
    """Return the ensemble risk-classification result for a given
    run_id, where run_id is an order_id (see plan doc for why
    ingestion_run_id doesn't apply here)."""
    try:
        order_id = int(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid run_id: {run_id}")
    return _classify_order(order_id)
