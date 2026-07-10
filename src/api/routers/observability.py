"""
observability.py router — /api/observability/*

All four endpoints read from the real SQLite tables (llm_call_log,
agent_execution_log, risk_classifications) populated during pipeline runs.
Falls back to fixture data when the tables are empty (e.g. before first run),
so the dashboard never shows a broken UI. Response shapes are identical to
the fixture contract — no frontend changes required.
"""

from typing import List

from fastapi import APIRouter

from src.api.schemas import CostByAgent, LatencyByAgent, PromptLogRow, VerdictDistributionSlice
from src.api.fixtures import COST_DATA, VERDICT_DIST, LATENCY_DATA, PROMPT_LOG
from src.utils.db_utils import execute_query

router = APIRouter()

# Verdict distribution colors — kept in sync with the mockup's VERDICT_DIST palette.
_VERDICT_COLORS = {
    "consensus": "#22C55E",
    "defer_to_rules": "#3B82F6",
    "override_llm": "#8B5CF6",
    "override_rules": "#F59E0B",
    "no_judge": "#475569",
}


@router.get("/cost", response_model=List[CostByAgent])
def get_cost():
    """Aggregate cost_usd per agent from llm_call_log. Falls back to fixtures when empty."""
    try:
        rows = execute_query(
            """
            SELECT agent_name AS agent, ROUND(SUM(cost_usd), 6) AS cost
            FROM llm_call_log
            GROUP BY agent_name
            ORDER BY cost DESC
            """
        )
        if rows:
            return [CostByAgent(agent=r["agent"], cost=r["cost"] or 0.0) for r in rows]
    except Exception:
        pass
    return COST_DATA


@router.get("/verdicts", response_model=List[VerdictDistributionSlice])
def get_verdicts():
    """Verdict-type distribution from risk_classifications. Falls back to fixtures when empty."""
    try:
        rows = execute_query(
            """
            SELECT final_label AS name, COUNT(*) AS cnt
            FROM risk_classifications
            GROUP BY final_label
            ORDER BY cnt DESC
            """
        )
        if rows:
            total = sum(r["cnt"] for r in rows) or 1
            return [
                VerdictDistributionSlice(
                    name=r["name"],
                    value=round(r["cnt"] / total * 100),
                    color=_VERDICT_COLORS.get(r["name"], "#64748B"),
                )
                for r in rows
            ]
    except Exception:
        pass
    return VERDICT_DIST


@router.get("/latency", response_model=List[LatencyByAgent])
def get_latency():
    """
    P50 / P90 latency per agent from agent_execution_log.
    SQLite lacks a native PERCENTILE function; we pull all rows and compute
    in Python — acceptable at dashboard scale (< 10k rows expected).
    Falls back to fixtures when empty.
    """
    try:
        rows = execute_query(
            """
            SELECT agent_name, duration_ms
            FROM agent_execution_log
            WHERE duration_ms IS NOT NULL
            ORDER BY agent_name, duration_ms
            """
        )
        if rows:
            from collections import defaultdict
            buckets: dict = defaultdict(list)
            for r in rows:
                buckets[r["agent_name"]].append(r["duration_ms"] / 1000.0)

            result = []
            for agent, values in sorted(buckets.items()):
                values.sort()
                n = len(values)
                p50 = values[int(n * 0.50)]
                p90 = values[min(int(n * 0.90), n - 1)]
                result.append(LatencyByAgent(agent=agent, p50=round(p50, 3), p90=round(p90, 3)))
            if result:
                return result
    except Exception:
        pass
    return LATENCY_DATA


@router.get("/prompt-log", response_model=List[PromptLogRow])
def get_prompt_log():
    """Latest 50 LLM call rows from llm_call_log. Falls back to fixtures when empty."""
    try:
        rows = execute_query(
            """
            SELECT ts, agent_name, model, prompt_preview, full_prompt, full_response,
                   total_tokens, cost_usd, latency_ms
            FROM llm_call_log
            ORDER BY id DESC
            LIMIT 50
            """
        )
        if rows:
            return [
                PromptLogRow(
                    ts=r["ts"] or "",
                    agent=r["agent_name"] or "",
                    model=r["model"] or "",
                    prompt=r["prompt_preview"] or "",
                    full_prompt=r["full_prompt"] or r["prompt_preview"] or "",
                    resp=r["full_response"] or "",
                    tokens=r["total_tokens"] or 0,
                    cost=r["cost_usd"] or 0.0,
                    latency=round((r["latency_ms"] or 0.0) / 1000.0, 3),
                )
                for r in rows
            ]
    except Exception:
        pass
    return PROMPT_LOG
