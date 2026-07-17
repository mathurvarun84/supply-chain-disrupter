"""
L2 News & Event Analysis Agent (gpt-4.1-mini).

Reads live news from SQLite (written by L1 ingestion), enriches with
semiconductor history + ChromaDB RAG, then classifies the disruption via LLM.

Does NOT call GDELT, RSS, or any live news API — that is L1's job.

Fallback chain when OpenAI is unavailable or fails:
  1. Rule-based FALLBACK_PARAMS (optionally bumped by live news volume)
  2. ChromaDB RAG via build_news_signals()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.rag.agent import build_news_signals
from src.agents.state import GlobalState, NewsAnalysisLLMOutput, NewsRiskSignal
from src.utils.db_utils import execute_query, fetch_recent_news
from src.utils.openai_utils import (
    MODEL_FAST,
    build_rag_context,
    call_openai_structured,
    format_semiconductor_signals,
    format_sqlite_record,
    has_openai_api_key,
)

logger = logging.getLogger(__name__)

NEWS_SYSTEM_PROMPT = """You are a supply-chain risk intelligence analyst specialising in
global electronics and semiconductor supply chains for procurement teams.

YOUR ROLE IN THE PIPELINE:
You classify disruption events and output the news/freight component (weight 0.15) for:
    Risk_Score = 0.40 × geo_risk  +  0.30 × supply_disruption
               + 0.15 × news_severity_component   ← YOUR KEY OUTPUT
               + 0.15 × defect_rate

SEVERITY CALIBRATION SCALE:
  0.90–1.00 : COVID-19 2020 global shutdown; Taiwan Strait blockade scenario
  0.75–0.89 : 2021 global chip shortage; 2023 HBM crunch
  0.65–0.74 : 2022 TSMC earthquake; Red Sea rerouting 2023-24
  0.50–0.64 : US export controls 2022; Shanghai lockdown 2022
  0.30–0.49 : Japan neon gas tightening; single-supplier delays
  0.10–0.29 : Routine port congestion (3-7d)
  0.00–0.09 : No disruption detected — normal operating conditions (0d)

expected_duration_days escalation matrix:
  0d/null → label unchanged | 2-3d → +1 tier | ≥4d → force CRITICAL

IMPORTANT — do not manufacture an event: user_severity_hint and the live news
signals below are real, pre-computed evidence for THIS run. When
user_severity_hint is near 0 and no live news rows are provided (or none are
relevant to this port/commodity), that means nothing is currently happening —
report severity/news_severity_component near 0 and expected_duration_days as
0 or null. Only report an elevated severity/duration when the evidence
(live news rows, RAG context, or a non-trivial user_severity_hint) actually
supports an active event. Escalating a quiet day to CRITICAL is a false
alarm, not a safe default.

FEW-SHOT EXAMPLES:

<example id="1" scenario="Red Sea Shipping Crisis">
<correct_response>
{
  "category": "logistics",
  "severity": 0.65,
  "affected_regions": ["Western Europe", "Southern Europe", "West Asia"],
  "affected_commodities": ["finished electronics", "display panels", "PCBs"],
  "news_severity_component": 0.72,
  "expected_duration_days": 180.0,
  "summary": "Houthi attacks forced Asia-Europe container traffic to reroute via Cape of Good Hope, adding 10-14 days transit and raising freight premiums 250%. European-sourced electronics face 6-month elevated logistics costs.",
  "signal_tags": ["red-sea", "shipping-route", "houthi", "logistics", "europe-asia"]
}
</correct_response>
</example>

<example id="2" scenario="TSMC Taiwan Earthquake">
<correct_response>
{
  "category": "weather",
  "severity": 0.72,
  "affected_regions": ["Eastern Asia", "Southeast Asia"],
  "affected_commodities": ["advanced logic chips (≤7nm)", "5G baseband chips", "AI accelerators"],
  "news_severity_component": 0.55,
  "expected_duration_days": 45.0,
  "summary": "Earthquake near TSMC Hsinchu triggered production halts and EUV recalibration, reducing advanced node wafer output 3-5% for the quarter. Premium smartphone SKUs face 6-8 week lead-time extensions.",
  "signal_tags": ["earthquake", "tsmc", "taiwan", "advanced-node", "semiconductor"]
}
</correct_response>
</example>

<example id="3" scenario="Live monitoring, no active event">
<open_meteo_data>
  user_severity_hint: 0.02 | live_news_rows: (No live news rows in SQLite — rely on RAG and calibration references.)
</open_meteo_data>
<correct_response>
{
  "category": "logistics",
  "severity": 0.03,
  "affected_regions": [],
  "affected_commodities": [],
  "news_severity_component": 0.02,
  "expected_duration_days": 0,
  "summary": "No live news signals or elevated risk indicators for this port/commodity at this time. Routine operating conditions — no active disruption to report.",
  "signal_tags": ["no-disruption", "routine-monitoring"]
}
</correct_response>
</example>

OUTPUT RULES:
- news_severity_component calibrated INDEPENDENTLY from severity
- summary must state disruption type, geography, recovery window, and procurement impact
- All fields required"""

# Calibrated rule-based scores used when the LLM path is unavailable.
FALLBACK_PARAMS: Dict[str, Dict[str, Any]] = {
    "earthquake": {"sev": 0.70, "comp": 0.55, "dur": 45.0, "cat": "weather"},
    "port_closure": {"sev": 0.60, "comp": 0.65, "dur": 14.0, "cat": "logistics"},
    "port closure": {"sev": 0.60, "comp": 0.65, "dur": 14.0, "cat": "logistics"},
    "chip_shortage": {"sev": 0.80, "comp": 0.50, "dur": 90.0, "cat": "raw_material"},
    "chip shortage": {"sev": 0.80, "comp": 0.50, "dur": 90.0, "cat": "raw_material"},
    "geopolitical": {"sev": 0.58, "comp": 0.48, "dur": 180.0, "cat": "geopolitical"},
    "extreme_weather": {"sev": 0.50, "comp": 0.40, "dur": 7.0, "cat": "weather"},
    "extreme weather": {"sev": 0.50, "comp": 0.40, "dur": 7.0, "cat": "weather"},
    "supplier_lockdown": {"sev": 0.65, "comp": 0.52, "dur": 30.0, "cat": "logistics"},
    "supplier lockdown": {"sev": 0.65, "comp": 0.52, "dur": 30.0, "cat": "logistics"},
    # "none" is the sentinel _derive_live_severity() (pipeline.py) sets when a
    # live run's ingestion sweep found no real weather/disaster/supply/news
    # signal for the port — a quiet day is a real outcome, not "unknown", and
    # must not fall through to _DEFAULT_FALLBACK's non-zero duration below.
    "none": {"sev": 0.0, "comp": 0.0, "dur": 0.0, "cat": "logistics"},
}

_DEFAULT_FALLBACK = {"sev": 0.40, "comp": 0.35, "dur": 7.0, "cat": "logistics"}


def _fetch_semiconductor_rows(year: Optional[Any]) -> List[dict]:
    """Step 2 helper — top-5 semiconductor_signals rows for the order year."""
    if year is None:
        return []
    try:
        rows = execute_query(
            "SELECT year, company, supply_disruption_index, export_control_level, "
            "known_disruption_event, known_severity FROM semiconductor_signals "
            "WHERE year = ? ORDER BY supply_disruption_index DESC LIMIT 5",
            (int(year),),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("semiconductor_signals fetch failed: %s", exc)
        return []


def _format_live_news_rows(rows: List[dict]) -> str:
    """Format L1-ingested news_signals rows for the LLM user message."""
    if not rows:
        return "(No live news rows in SQLite — rely on RAG and calibration references.)"
    lines = []
    for i, row in enumerate(rows[:20], start=1):
        lines.append(
            f"[{i}] {row.get('title', 'untitled')}\n"
            f"    region={row.get('detected_region')} "
            f"category={row.get('detected_category')} "
            f"published={row.get('published_at')}\n"
            f"    publisher={row.get('publisher')} query_tag={row.get('query_tag')}"
        )
    return "\n".join(lines)


def _build_news_user_message(
    disruption_type: str,
    affected_port: str,
    affected_route: str,
    severity_hint: float,
    shock_duration_days: int,
    recovery_window_days: int,
    record: dict,
    semiconductor_rows: List[dict],
    live_news_rows: List[dict],
    rag_context: str,
) -> str:
    """Step 4 helper — structured user message for the LLM call."""
    return f"""
═══════════════════════════════════════════════════════
SQLITE RECORD DATA (lite_master table — exact values)
═══════════════════════════════════════════════════════
{format_sqlite_record(record, "lite_master")}

═══════════════════════════════════════════════════════
SEMICONDUCTOR SIGNALS (semiconductor_signals table)
═══════════════════════════════════════════════════════
{format_semiconductor_signals(semiconductor_rows)}

═══════════════════════════════════════════════════════
LIVE NEWS SIGNALS (news_signals table — from L1 ingestion)
═══════════════════════════════════════════════════════
{_format_live_news_rows(live_news_rows)}

═══════════════════════════════════════════════════════
EVENT CONTEXT (from Scenario Analyzer)
═══════════════════════════════════════════════════════
  disruption_type          : {disruption_type}
  affected_port_or_hub     : {affected_port}
  affected_route           : {affected_route}
  user_severity_hint       : {severity_hint:.3f}
  shock_duration_days      : {shock_duration_days}
  recovery_window_days     : {recovery_window_days}

═══════════════════════════════════════════════════════
CHROMADB RAG CONTEXT (retrieved before this LLM call)
═══════════════════════════════════════════════════════
{rag_context if rag_context.strip() else "(No ChromaDB results — rely on calibration references.)"}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Classify this disruption event and return a NewsAnalysisLLMOutput.
Your news_severity_component (weight 0.15) feeds directly into the composite risk formula.
"""


def _llm_output_to_signals(llm_output: NewsAnalysisLLMOutput) -> List[NewsRiskSignal]:
    """Step 5 — primary signal plus up to 3 regional signals at 0.75× severity."""
    primary = NewsRiskSignal(
        source_id="llm-news-primary",
        category=llm_output.category,
        severity=llm_output.news_severity_component,
        summary=llm_output.summary,
        signal_tags=llm_output.signal_tags,
        expected_duration_days=llm_output.expected_duration_days,
    )
    region_signals = [
        NewsRiskSignal(
            source_id=f"llm-region-{i}",
            category=llm_output.category,
            severity=round(llm_output.news_severity_component * 0.75, 3),
            summary=f"Regional impact: {region}",
            signal_tags=[region.lower().replace(" ", "-"), llm_output.category],
            expected_duration_days=None,
        )
        for i, region in enumerate(llm_output.affected_regions[:3])
    ]
    return [primary] + region_signals


def _fallback_signals(
    disruption_type: str,
    live_news_count: int,
) -> List[NewsRiskSignal]:
    """Step 6 — rule-based fallback when LLM is unavailable.

    If more than 5 live news rows were ingested for this region, bump severity
    by +0.05 (capped at 0.85) to reflect confirmed media coverage.
    """
    params = FALLBACK_PARAMS.get(disruption_type, _DEFAULT_FALLBACK)
    comp = float(params["comp"])
    if live_news_count > 5:
        comp = min(0.85, round(comp + 0.05, 3))

    return [
        NewsRiskSignal(
            source_id="fallback-primary",
            category=params["cat"],
            severity=comp,
            summary=(
                f"Rule-based fallback for {disruption_type} "
                f"(LLM unavailable; live_news_rows={live_news_count})."
            ),
            signal_tags=[disruption_type.replace(" ", "-"), "fallback"],
            expected_duration_days=params["dur"],
        )
    ]


def news_event_analysis_agent(state: GlobalState) -> Dict[str, Any]:
    """
    L2 News & Event Analysis Agent.

    SQLite-first: reads news_signals written by L1, never calls live news APIs.
    LLM path: gpt-4.1-mini + ChromaDB RAG + SQLite context.
    Fallback: FALLBACK_PARAMS → build_news_signals().
    """
    metadata = state.event_metadata
    if metadata is None:
        raise ValueError("Event metadata is required for news analysis.")

    record = state.active_record or {}
    order_region = record.get("order_region") or record.get("port", "")

    # Step 1 — read live news from SQLite (L1 ingestion output).
    live_news_rows = fetch_recent_news(region=order_region or None, limit=20)
    live_news_count = len(live_news_rows)

    # Step 2 — semiconductor history for the order year.
    semiconductor_rows = _fetch_semiconductor_rows(record.get("year"))

    # Step 3 — ChromaDB RAG context (historical precedents, not live APIs).
    rag_context = build_rag_context([
        (
            f"supply chain {metadata.disruption_type} electronics semiconductor "
            "disruption historical precedent",
            4,
        ),
        (
            f"supply chain disruption {order_region} semiconductor risk impact recovery",
            3,
        ),
        (
            f"{metadata.disruption_type} logistics freight route disruption "
            "India electronics procurement",
            2,
        ),
    ])
    rag_chunk_count = rag_context.count("[") if rag_context else 0

    llm_used = False
    llm_output: Optional[NewsAnalysisLLMOutput] = None
    all_signals: List[NewsRiskSignal] = []

    # Step 4 — LLM classification (skipped when no API key).
    if has_openai_api_key():
        try:
            user_msg = _build_news_user_message(
                disruption_type=metadata.disruption_type,
                affected_port=metadata.affected_port,
                affected_route=metadata.affected_route,
                severity_hint=metadata.severity,
                shock_duration_days=metadata.shock_duration_days,
                recovery_window_days=metadata.recovery_window_days,
                record=record,
                semiconductor_rows=semiconductor_rows,
                live_news_rows=live_news_rows,
                rag_context=rag_context,
            )
            llm_output = call_openai_structured(
                system_prompt=NEWS_SYSTEM_PROMPT,
                user_message=user_msg,
                response_model=NewsAnalysisLLMOutput,
                model=MODEL_FAST,
                max_tokens=1024,
                run_id=state.run_id,
                agent_name="L2_news",
                trace=state.langfuse_trace,
                span=state.langfuse_span,
            )
            all_signals = _llm_output_to_signals(llm_output)
            llm_used = True
        except Exception as exc:
            logger.warning("L2 LLM failed — falling back: %s", exc)

    # Steps 6–7 — fallback chain: rule-based → ChromaDB RAG for unknown disruption types.
    if not all_signals:
        all_signals = _fallback_signals(metadata.disruption_type, live_news_count)
        if metadata.disruption_type not in FALLBACK_PARAMS:
            try:
                rag_signals = build_news_signals(metadata.disruption_type)
                if rag_signals:
                    all_signals = rag_signals
            except Exception as exc:
                logger.warning("L2 RAG fallback failed: %s", exc)

    cat = llm_output.category if llm_output else all_signals[0].category
    sev = llm_output.severity if llm_output else all_signals[0].severity
    comp = llm_output.news_severity_component if llm_output else all_signals[0].severity
    dur = (
        llm_output.expected_duration_days if llm_output else all_signals[0].expected_duration_days
    ) or 0

    log_msg = (
        f"L2: News {'(gpt-4.1-mini)' if llm_used else '(fallback)'} | "
        f"type={metadata.disruption_type} cat={cat} sev={sev:.3f} "
        f"comp={comp:.3f} dur={dur:.0f}d live_news={live_news_count} "
        f"rag_chunks={rag_chunk_count} signals={len(all_signals)}"
    )

    return {
        "news_signals": all_signals,
        "news_analysis_llm": llm_output,
        "agent_logs": state.agent_logs + [log_msg],
    }
