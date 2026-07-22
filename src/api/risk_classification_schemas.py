"""
risk_classification_schemas.py — response models for Screen 2 (Risk
Classification), GET /api/risk-classification/{run_id} and /latest.

Mirrors src/agents/state.py's RuleBasedSignal/DistilBERTSignal/LLMSignal/
JudgeVerdict field-for-field so the frontend gets full ensemble detail
instead of the flattened SignalResult shape in src/api/schemas.py (which
was the Day-1 fixture's shape and is kept as-is for other screens/tests).
Consumed by: src/frontend/src/app/TabRiskClassification.tsx via
types/riskClassification.ts + hooks/useRiskClassification.ts.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RuleSignalResponse(BaseModel):
    """Signal 1 — deterministic formula. Always present, never null."""
    composite_score: float
    geo_component: float
    supply_component: float
    freight_component: float
    defect_component: float
    base_label: RiskLevel
    escalated_label: RiskLevel
    escalated: bool
    duration_days: Optional[float] = None
    delivery_status_override: Optional[str] = None


class DistilBertSignalResponse(BaseModel):
    """Signal 2 — null-ish predicted_label ("N/A") when no fine-tuned model file is present."""
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    probability_distribution: Dict[str, float] = Field(default_factory=dict)
    model_source: str = "not-available-skipped"
    inference_ms: Optional[float] = None


class LlmSignalResponse(BaseModel):
    """Signal 3 — null when OPENAI_API_KEY is unset or the call failed."""
    predicted_label: Optional[RiskLevel] = None
    rationale: Optional[str] = None
    rag_citations: List[str] = Field(default_factory=list)
    rag_chunks_used: int = 0
    confidence_level: Optional[Literal["high", "medium", "low"]] = None
    primary_driver: Optional[
        Literal["geo", "supply", "freight", "defect", "delivery_status"]
    ] = None


class JudgeVerdictResponse(BaseModel):
    """LLM-as-Judge — null when the Judge call failed or was skipped (never crashes the endpoint)."""
    final_label: Optional[RiskLevel] = None
    verdict_type: Optional[
        Literal[
            "unanimous", "majority_rule", "override_distilbert",
            "override_llm", "defer_to_rules",
        ]
    ] = None
    reasoning: Optional[str] = None
    signals_agreed: Optional[bool] = None
    disagreement_explanation: Optional[str] = None


class RiskClassificationResponse(BaseModel):
    """
    Full Screen 2 payload for one order_id (exposed as run_id).

    final_label/final_critical_flag/slack_should_fire are ALWAYS derived
    server-side (final_critical_flag = final_label == "CRITICAL") — never
    taken from a raw LLM/judge field, per the escalation-guard contract in
    docs/ARCHITECTURE.md.
    """
    run_id: str
    order_id: int
    mode: Literal["live", "replay"]
    rule_signal: RuleSignalResponse
    distilbert_signal: DistilBertSignalResponse
    llm_signal: LlmSignalResponse
    judge_verdict: Optional[JudgeVerdictResponse] = None
    final_label: RiskLevel
    final_critical_flag: bool
    slack_should_fire: bool
    threshold: float = 0.75
    from_cache: bool
    # Same winning SKU_id L4 resolved for this run -- shown next to
    # Forecast/Simulation/Mitigation's own sku_id so all four agents' output
    # can be visually confirmed as running on the same SKU. None on the
    # legacy order_id cache path for rows predating the SKU crosswalk.
    sku_id: Optional[str] = None
    # Top-level mirror of rule_signal.duration_days -- the canonical value
    # threaded to L5/L6/L7 via ForecastHandoff.duration_days -- so the
    # frontend can read the same field name across all four response types
    # instead of reaching into rule_signal just for Screen 2.
    impact_duration_days: Optional[float] = None
