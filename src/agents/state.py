# src/agents/state.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class EventMetadata(BaseModel):
    disruption_type: str
    affected_port: str
    affected_route: str
    severity: float
    shock_duration_days: int
    recovery_window_days: int
    synthetic_ratio: float
    simulation_trials: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Monte Carlo trial count for L6 impact simulation.",
    )


class NewsRiskSignal(BaseModel):
    source_id: str
    category: str
    severity: float
    summary: str
    signal_tags: List[str]
    expected_duration_days: Optional[float] = Field(
        None,
        description="LLM duration estimate in days — drives Risk Classifier escalation matrix",
    )


class ForecastResult(BaseModel):
    # Core fields (required — backward-compatible with prior L5 stub)
    prophet_forecast: List[Dict[str, Any]] = Field(default_factory=list)  # legacy alias
    expected_drop_pct: float

    # v4 fields
    demand_forecast: List[Dict[str, Any]] = Field(default_factory=list)   # from selected model
    model_selected: str = "prophet"
    model_comparison_scores: Dict[str, Any] = Field(default_factory=dict)

    # Extended fields from DemandForecastingAgent v3+ (all optional)
    sku_id: Optional[str] = None
    regressors_used: List[str] = Field(default_factory=list)
    regressor_selection_method: str = "backtest_ablation"
    stockout_prob: Optional[float] = None
    mape_prophet_trend_only: Optional[float] = None
    mape_prophet_selected: Optional[float] = None
    mape_dataset_baseline_avg: Optional[float] = None
    mape_dataset_ai_avg: Optional[float] = None
    mape_improvement_pct_vs_dataset_baseline: Optional[float] = None
    disruption_scenario: Optional[Dict[str, Any]] = None
    forecast_agent_logs: List[str] = Field(default_factory=list)


class ForecastHandoff(BaseModel):
    """
    Snapshot L4 hands to L5 so L5 never needs to re-query lite_master for
    basic context. Sourced from the SINGLE winning record select_forecast_sku()
    chose -- never from a non-winning candidate. All fields except sku_id,
    risk_score_composite, and risk_label are Optional because they mirror
    lite_master columns that can themselves be null/missing on some rows.
    """
    sku_id: str
    order_id: Optional[Any] = None
    product_name: Optional[str] = None
    category_name: Optional[str] = None
    order_date: Optional[str] = None
    unit_price_usd: Optional[float] = None
    sales_usd: Optional[float] = None
    risk_score_composite: float
    risk_label: str
    candidates_considered: int = 1   # audit trail: how many records were in play for this event


class SimulationResult(BaseModel):
    stockout_probability_pct: float
    expected_inventory_gap_pct: float
    alternate_route: Optional[str]
    stockout_probability_p10: Optional[float] = None
    stockout_probability_p90: Optional[float] = None
    days_to_stockout_p50: Optional[float] = None
    days_to_stockout_p10: Optional[float] = None
    days_to_stockout_p90: Optional[float] = None
    revenue_impact_usd_p50: Optional[float] = None
    revenue_impact_usd_p10: Optional[float] = None
    revenue_impact_usd_p90: Optional[float] = None
    trials_run: int = 0
    model_version: str = "mc_v1"
    revenue_impact_samples: List[float] = Field(default_factory=list)


class MitigationAction(BaseModel):
    summary: str
    recommendations: List[str]
    cost_delta: str
    urgency: str = Field("HIGH")
    rag_citations: List[str] = Field(default_factory=list)
    india_sourcing_recommendations: List[str] = Field(default_factory=list)


# ── LLM output models (structured OpenAI responses) ──────────────────────────

class WeatherRiskLLMOutput(BaseModel):
    """L3 LLM output — supply-chain interpretation of Open-Meteo numeric data."""

    event_classification: Literal["extreme", "severe", "moderate", "minor", "clear"] = Field(
        ...,
        description=(
            "Supply-chain severity tier for current weather at this hub. "
            "extreme: typhoon/earthquake forcing fab closure (>72h). "
            "severe: major storm causing 24-72h logistics delays. "
            "moderate: contingency monitoring required. "
            "minor: marginal impact, normal ops with caution. "
            "clear: no weather risk."
        ),
    )
    geo_risk_component: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Enhanced geo risk for composite formula (weight 0.40). "
            "This value overrides the raw numeric Open-Meteo severity."
        ),
    )
    affected_semiconductor_hubs: List[str] = Field(
        ...,
        description=(
            "Impacted hubs from: Hsinchu, Tainan, Osaka, Austin, Shanghai, "
            "Singapore, Rotterdam, Incheon, Penang, Ho_Chi_Minh_City, Shenzhen, Chennai."
        ),
    )
    supply_chain_narrative: str = Field(
        ...,
        description=(
            "2-3 sentences naming the specific fab or logistics node at risk, "
            "estimated delay in days if severe+, and which product category is most exposed."
        ),
    )
    rag_escalation_warranted: bool = Field(
        ...,
        description=(
            "True when geo_risk_component >= 0.65 — signals Risk Classifier to query "
            "ChromaDB for historical weather precedents at this hub."
        ),
    )


class NewsAnalysisLLMOutput(BaseModel):
    """L2 LLM output — structured event classification from disruption type + RAG context."""

    category: Literal["weather", "geopolitical", "logistics", "raw_material", "demand_shock"] = Field(
        ...,
        description=(
            "geopolitical: export controls, sanctions, trade wars. "
            "logistics: port closures, shipping route disruptions. "
            "raw_material: rare earth, wafer supply constraints. "
            "demand_shock: AI surges, inventory gluts. "
            "weather: natural disasters."
        ),
    )
    severity: float = Field(..., ge=0.0, le=1.0, description="Event severity 0-1.")
    affected_regions: List[str] = Field(
        ...,
        description="DataCo region names e.g. Eastern Asia, Western Europe.",
    )
    affected_commodities: List[str] = Field(
        ...,
        description="Specific product classes e.g. advanced logic chips, DRAM memory.",
    )
    news_severity_component: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Freight/logistics proxy for composite formula (weight 0.15).",
    )
    expected_duration_days: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Days until primary disruption resolves. Drives escalation matrix. "
            "0 or null when the evidence does not support an active disruption "
            "— do not invent a positive duration when there is none."
        ),
    )
    summary: str = Field(
        ...,
        description="2-3 sentences: disruption type, geography, recovery window, procurement impact.",
    )
    signal_tags: List[str] = Field(
        ...,
        description="3-6 lowercase hyphenated tags e.g. ['red-sea', 'logistics'].",
    )


class RiskClassifierLLMEnhancement(BaseModel):
    """L4 LLM output — narrative layer added ON TOP of rule-based classification."""

    primary_risk_driver: Literal["geo", "supply", "freight", "defect"] = Field(
        ...,
        description="Component with highest normalised contribution to composite score.",
    )
    enhanced_rationale: str = Field(
        ...,
        description="3-4 sentences for procurement manager citing RAG and component values.",
    )
    evaluator_one_liner: str = Field(
        ...,
        description="≤20 words for Streamlit dashboard risk card.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="high: composite > 0.80 OR delivery_status override applied.",
    )


class MitigationLLMOutput(BaseModel):
    """L7 LLM output — ranked mitigation plan with India sourcing and RAG citations."""

    summary: str = Field(
        ...,
        description="2-3 sentence executive summary for dashboard.",
    )
    ranked_actions: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "1-5 specific, procurement-actionable items, most urgent first. Do not pad with "
            "low-value filler to hit a minimum — a genuinely low-risk scenario may warrant "
            "only 1-2 actions."
        ),
    )
    cost_estimate: str = Field(
        ...,
        description="Format: '<LEVEL>: <reason>'. HIGH | MEDIUM | LOW.",
    )
    urgency: Literal["IMMEDIATE", "HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="IMMEDIATE=CRITICAL label, HIGH=HIGH label, etc.",
    )
    rag_citations: List[str] = Field(
        default_factory=list,
        description=(
            "Citations naming sources actually present in the provided RAG context. Empty list "
            "is valid when no retrieved chunk is a genuine fit — never invent a source; fabricated "
            "citations are filtered out downstream against the real retrieved sources."
        ),
    )
    india_sourcing_recommendations: List[str] = Field(
        default_factory=list,
        description=(
            "Named India/ASEAN option(s) that genuinely match the affected commodity/product "
            "category in the RAG context. Empty list when no real fit exists — never force a "
            "facility or scheme onto an unrelated commodity."
        ),
    )


# ── Ensemble signal models (L4 three-signal + judge) ─────────────────────────

class RuleBasedSignal(BaseModel):
    """Signal 1 — deterministic composite formula + overrides."""

    composite_score: float
    geo_component: float
    supply_component: float
    freight_component: float
    defect_component: float
    base_label: str
    escalated_label: str
    escalated: bool
    duration_days: Optional[float]
    delivery_status_override: Optional[str] = None


class DistilBERTSignal(BaseModel):
    """Signal 2 — Fine-tuned DistilBERT classifier (~20ms CPU)."""

    predicted_label: str
    confidence: float
    probability_distribution: Dict[str, float]
    model_source: str
    inference_ms: float


class LLMSignal(BaseModel):
    """Signal 3 — GPT-4o + two-stage RAG."""

    predicted_label: str
    rationale: str
    rag_citations: List[str]
    rag_chunks_used: int
    confidence_level: Literal["high", "medium", "low"]
    primary_driver: Literal["geo", "supply", "freight", "defect", "delivery_status"]


class JudgeVerdict(BaseModel):
    """LLM-as-Judge final decision after seeing all 3 signals."""

    final_label: str
    verdict_type: Literal[
        "unanimous",
        "majority_rule",
        "override_distilbert",
        "override_llm",
        "defer_to_rules",
    ]
    reasoning: str
    signals_agreed: bool
    disagreement_explanation: Optional[str] = None
    final_critical_flag: bool


class RiskClassificationResult(BaseModel):
    """Full audit trail for one risk classification run."""

    mode: str
    composite_score: float = Field(..., ge=0.0, le=1.0)
    geo_component: float
    supply_component: float
    freight_component: float
    defect_component: float
    duration_days: Optional[float]
    base_label: str
    final_label: str
    escalated: bool
    rag_citations: List[str] = Field(default_factory=list)
    rationale: str
    critical_flag: bool
    llm_enhanced_rationale: Optional[str] = None
    llm_evaluator_one_liner: Optional[str] = None
    llm_primary_driver: Optional[str] = None
    llm_confidence: Optional[str] = None
    rule_signal: Optional[RuleBasedSignal] = None
    distilbert_signal: Optional[DistilBERTSignal] = None
    llm_signal: Optional[LLMSignal] = None
    judge_verdict: Optional[JudgeVerdict] = None
    sku_id: Optional[str] = None  # from active_record.sku_id; None if the
                                   # source workbook predates the crosswalk
                                   # or used the strict-1:1 mapping variant


class GlobalState(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    config: Optional[Dict[str, Any]] = None
    active_record: Optional[Dict[str, Any]] = None
    # Every record a single event could plausibly implicate (region-wide demo
    # scenarios, etc). Normally just [active_record] -- see select_forecast_sku()
    # in risk_classifier_agent, which picks exactly one before L4's ensemble runs.
    candidate_records: List[Dict[str, Any]] = Field(default_factory=list)
    forecast_handoff: Optional[ForecastHandoff] = None  # L4's context snapshot for L5
    ingestion_run_id: Optional[str] = None  # UUID from L1; links state to live_news_ingest / live_weather_ingest rows
    news_signals: List[NewsRiskSignal] = Field(default_factory=list)
    live_weather_severity: Optional[float] = None
    risk_classification: Optional[RiskClassificationResult] = None
    forecast_result: Optional[ForecastResult] = None
    simulation_result: Optional[SimulationResult] = None
    mitigation_action: Optional[MitigationAction] = None
    # Structured per-query RAG trace from L7's build_mitigation_context_structured()
    # call — always 3 entries (historical/export-control/india) when the LLM+RAG
    # path ran; empty when it didn't (no OpenAI key, rule-based fallback used).
    # Consumed by pipeline_bridge.persist_mitigation_output() for the Screen 4
    # RAG Query Trace panel; not part of MitigationAction because it's retrieval
    # metadata, not an agent recommendation.
    mitigation_rag_trace: List[Dict[str, Any]] = Field(default_factory=list)
    agent_logs: List[str] = Field(default_factory=list)
    news_analysis_llm: Optional[NewsAnalysisLLMOutput] = None
    weather_risk_llm: Optional[WeatherRiskLLMOutput] = None
    risk_enhancement_llm: Optional[RiskClassifierLLMEnhancement] = None
    mitigation_llm: Optional[MitigationLLMOutput] = None
    judge_verdict: Optional[JudgeVerdict] = None

    # Observability fields — set by the orchestrator before each agent call.
    # exclude=True prevents non-serializable Langfuse SDK objects from leaking
    # into model_dump_json() calls (API responses, agent_logs serialization, etc.)
    run_id: Optional[str] = None
    langfuse_trace: Optional[Any] = Field(default=None, exclude=True)
    langfuse_span: Optional[Any] = Field(default=None, exclude=True)

    @property
    def risk_label(self) -> Optional[str]:
        """Deprecated shim — read risk_classification.final_label instead."""
        return self.risk_classification.final_label if self.risk_classification else None

    @property
    def risk_score_composite(self) -> Optional[float]:
        """Deprecated shim — read risk_classification.composite_score instead."""
        return self.risk_classification.composite_score if self.risk_classification else None
