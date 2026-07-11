"""
observability.py — Dual-write observability for the L1–L7 LangGraph pipeline.

All Langfuse operations are best-effort and wrapped in try/except — a Langfuse
outage or misconfiguration NEVER raises or blocks the pipeline. SQLite writes
(agent_execution_log, llm_call_log) are unconditional: they happen regardless
of whether Langfuse is enabled or reachable.

Kill switch: set LANGFUSE_ENABLED=false (or leave unset) to disable all
Langfuse I/O while keeping SQLite logging intact.

Cost table cross-reference: MODEL_PRICING must stay in sync with
MODEL_REASONING / MODEL_FAST in src/utils/openai_utils.py. Update both in
the same PR if model names change.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kill switch & client singleton
# ---------------------------------------------------------------------------

def observability_enabled() -> bool:
    """True only when LANGFUSE_ENABLED=true AND both API keys are set."""
    return (
        os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


@lru_cache(maxsize=1)
def _get_langfuse_client():
    """Process-lifetime Langfuse client. Returns None when disabled or on init failure."""
    if not observability_enabled():
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        logger.warning("Langfuse client init failed, disabling for this process: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

# USD per 1,000 tokens. Source: OpenAI pricing page (July 2026).
# Documented here so the evaluation report can cite these figures directly
# rather than treating them as magic numbers.
# Cross-reference: MODEL_REASONING = "gpt-4o", MODEL_FAST = "gpt-4.1-mini"
# in src/utils/openai_utils.py — update both files together if models change.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":       {"input": 0.0025, "output": 0.010},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
}


def calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for one LLM call from the static pricing table."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logger.warning("No pricing entry for model=%s, cost logged as 0.0", model)
        return 0.0
    return round(
        (input_tokens / 1000) * pricing["input"]
        + (output_tokens / 1000) * pricing["output"],
        6,
    )


def build_langfuse_trace_url(run_id: str) -> Optional[str]:
    """Return a direct Langfuse trace URL for a run_id, or None when disabled."""
    if not observability_enabled():
        return None
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return f"{host}/trace/{run_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def pipeline_trace(
    run_id: str,
    mode: str,
    source_type: str = "unknown",
    metadata: Optional[dict] = None,
) -> Generator[Any, None, None]:
    """
    Open one Langfuse trace for an entire L1–L7 run. Yields the trace handle
    (or None when disabled). Always yields — never raises.

    Call client.flush() in finally so events aren't lost when the process
    exits shortly after a FastAPI BackgroundTask completes.
    """
    client = _get_langfuse_client()
    trace = None
    if client is not None:
        try:
            trace = client.trace(
                id=run_id,
                name="supply_chain_pipeline_run",
                metadata={"mode": mode, "source_type": source_type, **(metadata or {})},
                tags=[mode, source_type],
            )
        except Exception as exc:
            logger.warning("Langfuse trace open failed for run_id=%s: %s", run_id, exc)
    try:
        yield trace
    finally:
        if client is not None and trace is not None:
            try:
                client.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)


@contextmanager
def agent_span(
    trace: Any,
    run_id: str,
    agent_name: str,
    input_summary: Optional[dict] = None,
) -> Generator[Any, None, None]:
    """
    Wrap one agent's execution (any of L1–L7, LLM or non-LLM).

    SQLite write to agent_execution_log is UNCONDITIONAL — happens regardless
    of Langfuse availability. Langfuse span is best-effort only.

    Re-raises real agent exceptions so _run_optional() and the caller's
    error handling remain unchanged — observability adds visibility, not
    different failure semantics.
    """
    from src.utils.db_utils import insert_agent_execution, update_agent_execution

    started = time.monotonic()
    started_iso = _utcnow_iso()
    try:
        insert_agent_execution(
            run_id=run_id, agent_name=agent_name,
            status="Running", started_at=started_iso,
        )
    except Exception as exc:
        logger.warning("agent_execution_log insert failed for %s: %s", agent_name, exc)

    span = None
    if trace is not None:
        try:
            span = trace.span(name=agent_name, input=input_summary or {})
        except Exception as exc:
            logger.warning("Langfuse span open failed for %s: %s", agent_name, exc)

    status = "Complete"
    error_message = None
    try:
        yield span
    except Exception as exc:
        status = "Failed-Fallback"
        error_message = str(exc)
        raise
    finally:
        duration_ms = (time.monotonic() - started) * 1000
        try:
            update_agent_execution(
                run_id=run_id,
                agent_name=agent_name,
                status=status,
                completed_at=_utcnow_iso(),
                duration_ms=duration_ms,
                error_message=error_message,
                langfuse_trace_id=(getattr(trace, "id", None) if trace is not None else None),
                langfuse_span_id=(getattr(span, "id", None) if span is not None else None),
            )
        except Exception as exc:
            logger.warning("agent_execution_log update failed for %s: %s", agent_name, exc)
        if span is not None:
            try:
                span.end(output={"status": status, "duration_ms": duration_ms})
            except Exception as exc:
                logger.warning("Langfuse span close failed for %s: %s", agent_name, exc)


def record_llm_generation(
    trace: Any,
    span: Any,
    *,
    run_id: str,
    agent_name: str,
    model: str,
    system_prompt: str,
    user_message: str,
    parsed_output: Any,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str = "success",
    retry_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """
    Record one completed (or failed) OpenAI call.

    Always writes to llm_call_log (SQLite). Best-effort writes a nested
    Langfuse generation under the owning agent span.

    Called from call_openai_structured() in openai_utils.py — not from
    agent files directly — so a single instrumentation point covers all
    four LLM agents (L2, L3, L4×2).
    """
    from src.utils.db_utils import insert_llm_call_log

    cost = calculate_cost_usd(model, input_tokens, output_tokens)
    generation_id = str(uuid.uuid4())

    try:
        insert_llm_call_log(
            run_id=run_id,
            agent_name=agent_name,
            model=model,
            prompt_preview=(user_message or "")[:200],
            full_prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_message}",
            full_response=str(parsed_output) if parsed_output is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            status=status,
            retry_count=retry_count,
            error_message=error_message,
            langfuse_trace_id=(getattr(trace, "id", None) if trace is not None else None),
            langfuse_generation_id=generation_id,
        )
    except Exception as exc:
        logger.warning("llm_call_log insert failed for %s: %s", agent_name, exc)

    if span is not None:
        try:
            span.generation(
                id=generation_id,
                name=f"{agent_name}_llm_call",
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                output=str(parsed_output) if parsed_output is not None else None,
                usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
                metadata={"cost_usd": cost, "retry_count": retry_count, "status": status},
                level="ERROR" if status == "failed_fallback" else "DEFAULT",
                status_message=error_message,
            )
        except Exception as exc:
            logger.warning("Langfuse generation record failed for %s: %s", agent_name, exc)
