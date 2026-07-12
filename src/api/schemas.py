from typing import Any, List, Optional, Literal, Dict
from pydantic import BaseModel

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


class RankedAction(BaseModel):
    rank: int
    text: str
    citations: List[str]


class MitigationResponse(BaseModel):
    run_id: str
    urgency: Literal["ROUTINE", "ELEVATED", "IMMEDIATE"]
    ranked_actions: List[RankedAction]
    rag_query_trace: List[str]
    india_sourcing_recommendations: List[str]
    slack_preview: str
    cost_delta_usd: float


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
    name: str
    dir: Literal["input", "output"]
    agent: str
    pass_count: int
    fail_count: int
    reason: str


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
