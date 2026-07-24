from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

AgentStatus = Literal["Idle", "Running", "Complete", "Skipped-Optional", "Failed-Fallback"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SourceType = Literal["LIVE", "DEMO-INJECTED", "REPLAY"]


class AgentState(BaseModel):
    id: str
    name: str
    status: AgentStatus
    duration_ms: Optional[float] = None


class PipelineStatus(BaseModel):
    run_id: str
    source_type: SourceType
    agents: List[AgentState]
    last_ingested_at: Optional[str] = None
    openai_status: Literal["connected", "disconnected"] = "connected"
    langfuse_trace_url: Optional[str] = None
    is_complete: bool = False
    current_phase: Optional[str] = None


DemoScenarioId = Literal[
    "taiwan_earthquake", "red_sea_crisis", "guardrail_demo", "clean_baseline"
]


class PipelineRunRequest(BaseModel):
    mode: Literal["live", "demo", "replay"]
    demo_scenario_id: Optional[DemoScenarioId] = None
    replay_run_id: Optional[str] = None


class PipelineRunResponse(BaseModel):
    run_id: str
    mode: Literal["live", "demo", "replay"]
    accepted_at: str


class NewsItem(BaseModel):
    headline: str
    source: str
    tag: str
    time: str
    score: float


class NewsGroup(BaseModel):
    group: str
    items: List[NewsItem]


class WeatherCity(BaseModel):
    name: str
    flag: str
    wind: float
    precip: float
    temp: float
    icon: str
    severity: float
    trigger: bool


class LogLine(BaseModel):
    level: str
    text: str
    tab: int


class GanttRow(BaseModel):
    id: str
    start: float
    dur: float
    color: str


class SignalResult(BaseModel):
    label: RiskLevel
    detail: Dict[str, float] = {}
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    citations: List[str] = []


class RiskClassification(BaseModel):
    run_id: str
    verdict_type: Literal[
        "unanimous", "majority_rule", "override_distilbert", "override_llm", "defer_to_rules"
    ]
    composite_score: float
    threshold: float
    rule_signal: SignalResult
    distilbert_signal: SignalResult
    llm_signal: SignalResult
    judge_text: str
    slack_should_fire: bool


class ForecastPoint(BaseModel):
    day: str
    baseline: float
    adjusted: float


class ForecastResponse(BaseModel):
    run_id: str
    category: str
    categories: List[str]
    series: List[ForecastPoint]
    # L4's canonical duration_days, threaded through ForecastHandoff --
    # shown alongside Simulation/Mitigation's own copy so the UI can
    # confirm all three agents used the same disruption length.
    impact_duration_days: Optional[float] = None


class ForecastWeekPoint(BaseModel):
    """One week in a per-SKU 5-week Prophet forecast (L5 DemandForecastingAgent v3)."""
    week_start: str
    demand_baseline: float
    demand_disrupted: float


class SkuForecastResponse(BaseModel):
    """Full L5 forecast response for a single SKU."""
    sku_id: str
    forecast_horizon_weeks: int
    regressors_used: List[str]
    regressor_selection_method: str
    model_selected: str = "prophet"
    model_comparison_scores: Dict[str, Any] = {}
    expected_drop_pct: float
    stockout_prob: Optional[float]
    mape_prophet_trend_only: Optional[float]
    mape_prophet_selected: Optional[float]
    mape_dataset_baseline_avg: Optional[float]
    mape_dataset_ai_avg: Optional[float]
    mape_improvement_pct_vs_dataset_baseline: Optional[float]
    disruption_scenario: Optional[Dict[str, Any]]
    demand_forecast: List[ForecastWeekPoint] = []
    prophet_forecast: List[ForecastWeekPoint] = []  # legacy alias
    generated_at_utc: Optional[str] = None


class ModelScore(BaseModel):
    rmse: float
    rmsle: float
    smape: float
    mape: float
    latency_sec: float


class ModelComparisonResponse(BaseModel):
    """Backtest comparison of candidate forecast models for one SKU (L5 DemandForecastingAgent).

    Mirrors DemandForecastingAgent.get_model_comparison_chart_data(), previously
    only exposed via the standalone Flask dashboard (src/forecast_dashboard.py).
    """
    sku_id: str
    labels: List[str]
    actual: List[float]
    prophet: List[float]
    sarimax: List[float]
    timegpt: Optional[List[float]] = None
    timegpt_status: str
    winner: str
    scores: Dict[str, ModelScore]


class SimulationBucket(BaseModel):
    range: str
    count: int


class SimulationResponse(BaseModel):
    run_id: str
    p10: float
    p50: float
    p90: float
    revenue_at_risk_usd: float
    alternate_route: str
    histogram: List[SimulationBucket]
    # Real SimulationResult fields (run_monte_carlo() always computes these);
    # Optional because pre-fix demo/fixture rows and the heuristic fallback
    # path don't populate them.
    revenue_at_risk_p10_usd: Optional[float] = None
    revenue_at_risk_p90_usd: Optional[float] = None
    days_to_stockout_p10: Optional[float] = None
    days_to_stockout_p50: Optional[float] = None
    days_to_stockout_p90: Optional[float] = None
    # Same winning SKU_id L4/L5 resolved for this run — shown next to
    # Forecast/Mitigation's own sku_id so all three agents' output can be
    # visually confirmed as running on the same SKU.
    sku_id: Optional[str] = None
    # Disruption length this simulation's trials actually used -- L4's
    # canonical duration_days when available (see priors.py's
    # shock_duration resolution), so the UI can confirm L6 used the same
    # duration as Forecast/Mitigation.
    impact_duration_days: Optional[float] = None


class MitigationCitation(BaseModel):
    """One RAG-grounded citation, split from the agent's "<collection>: <source_file>"
    string — see src.utils.db_utils._parse_citation."""
    source_file: str
    collection: str


class RankedAction(BaseModel):
    rank: int
    text: str
    action_type: Literal["INVENTORY", "ROUTING", "SOURCING", "INDIA-SOURCING", "MONITOR", "FINANCIAL"]
    citations: List[MitigationCitation] = Field(default_factory=list)


class RagTraceChunk(BaseModel):
    source_file: Optional[str] = None
    collection: Optional[str] = None
    similarity_score: Optional[float] = None
    snippet: Optional[str] = None


class RagTraceQuery(BaseModel):
    """One row in the expandable RAG Query Trace panel."""
    query_name: Literal["historical_disruption_lookup", "export_control_check", "india_sourcing_query"]
    query_text: str
    fired: bool
    fire_condition: str
    retrieved_chunks: List[RagTraceChunk] = Field(default_factory=list)


class MitigationResponse(BaseModel):
    run_id: str
    risk_level: RiskLevel
    summary: Optional[str] = None
    urgency: Literal["LOW", "MEDIUM", "HIGH", "IMMEDIATE"]
    mitigation_window: Optional[str] = None
    ranked_actions: List[RankedAction]
    rag_query_trace: List[RagTraceQuery]
    india_sourcing_recommendations: List[str]
    slack_alert_fired: bool
    slack_preview: Optional[str] = None
    cost_delta: Optional[str] = None
    cost_delta_usd: Optional[float] = None
    # Same winning SKU_id L4/L5/L6 resolved for this run — shown next to
    # Forecast/Simulation's own sku_id so all three agents' output can be
    # visually confirmed as running on the same SKU.
    sku_id: Optional[str] = None
    # Numeric twin of mitigation_window's formatted string -- same L4
    # canonical duration_days source, for a consistent badge across panels.
    impact_duration_days: Optional[float] = None


class CostByAgent(BaseModel):
    agent: str
    cost: float


class VerdictDistributionSlice(BaseModel):
    name: str
    value: int
    color: str


class LatencyByAgent(BaseModel):
    agent: str
    p50: float
    p90: float
    p99: float


class PromptLogRow(BaseModel):
    ts: str
    agent: str
    model: str
    prompt: str          # truncated preview for the table row
    resp: str            # full serialized response
    full_prompt: Optional[str] = None  # system + user, for expand view
    tokens: int
    cost: float
    latency: float


class GuardrailEvent(BaseModel):
    """Aggregated view of guardrail_events (doc §5.2), grouped by
    (guardrail_name, direction, agent_name). Consumed by the Guardrails
    sub-tab's Activity table in TabObservability.tsx."""

    name: str
    dir: Literal["input", "output", "execution"]
    agent: str
    pass_count: int
    fail_count: int
    last_reason: str


class RagasScore(BaseModel):
    metric: str
    score: float
    threshold: float
    passed: bool


class CorpusHealth(BaseModel):
    name: str
    docs: int
    real: int
    synth: int
    last_ingested_at: str


class GoldQARow(BaseModel):
    question: str
    ground_truth: str
    match: bool
    source_collection: Optional[str] = None
    source_chunk_id: Optional[str] = None
    query_style: Literal["agent_pattern", "natural_question"] = "natural_question"


class DatabaseStatus(BaseModel):
    database_exists: bool
    tables: Optional[Dict[str, int]] = None
    date_range: Optional[str] = None
    categories: Optional[List[str]] = None
    unique_products: Optional[int] = None
    size_mb: Optional[float] = None


class AdminJobStatus(BaseModel):
    status: Literal["idle", "running", "complete", "failed"]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AdminJobTriggerResponse(BaseModel):
    status: Literal["started", "skipped_already_running"]
    triggered_at: str


class AdminStatusResponse(BaseModel):
    database: DatabaseStatus
    db_job: AdminJobStatus
    rag_job: AdminJobStatus
    corpus: List[CorpusHealth]


class TableSummary(BaseModel):
    name: str
    row_count: int
    column_count: int


class TableListResponse(BaseModel):
    tables: List[TableSummary]


class TableRowsResponse(BaseModel):
    table_name: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    total_pages: int
