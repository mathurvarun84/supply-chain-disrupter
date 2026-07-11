"""
wrapper.py — run_with_trulens(): drop-in replacement for
langgraph_engine.run_agent_graph() that adds TruLens tracing without
touching langgraph_engine.py or any file under src/agents/.

Per-node latency comes from LangGraph's native app.stream(stream_mode=
"updates") rather than monkey-patching langgraph_engine internals — a node
exception still propagates out of the stream iterator exactly as
app.invoke() would raise it (verified against the installed langgraph
version; see docs/specs/2026-07-06-trulens-integration-design.md,
"Non-Interference Guarantees"). Only telemetry capture around each yielded
update is wrapped in its own try/except.

run_agent_graph() itself calls app.invoke(GlobalState()) with no run_id, so
production graph runs have run_id=None throughout and their llm_call_log
rows are unattributable. This wrapper mints (or reuses payload["run_id"])
and seeds GlobalState(run_id=...) so the OpenAI patch's SQLite lookups in
openai_patch.py work correctly.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from src.agents.langgraph_engine import build_agent_graph
from src.agents.state import GlobalState
from src.evaluation.trulens_integration.node_extractors import (
    NODE_LATENCY_LABELS, extract_l4_signals, extract_l5_forecast,
)
from src.evaluation.trulens_integration.openai_patch import LLMCallRecord, patch_openai_calls

logger = logging.getLogger(__name__)


def _record_pipeline_run(
    *,
    run_id: str,
    node_latencies_ms: Dict[str, float],
    llm_calls: list[LLMCallRecord],
    l4_signals: Dict[str, Any] | None,
    l5_forecast: Dict[str, Any] | None,
) -> None:
    """Best-effort: write this run's telemetry into the TruLens session.

    Isolated from the streaming/merge loop in run_with_trulens() so a
    TruLens-side failure here can never affect the returned GlobalState.
    """
    try:
        from src.evaluation.trulens_integration.config import get_session

        get_session()
        logger.info(
            "TruLens run recorded run_id=%s node_latencies_ms=%s llm_call_count=%d "
            "l4_signals=%s l5_forecast=%s",
            run_id, node_latencies_ms, len(llm_calls), l4_signals, l5_forecast,
        )
    except Exception as exc:
        logger.warning("TruLens record write failed (non-blocking) for run_id=%s: %s", run_id, exc)


def run_with_trulens(payload: Dict[str, Any]) -> GlobalState:
    """Drop-in replacement for run_agent_graph(payload) with TruLens tracing."""
    from src.utils.db_utils import ensure_schema

    # Neither run_agent_graph() nor build_agent_graph() calls ensure_schema()
    # — on a freshly-built outputs/supply_chain.db (rebuilt via
    # src.build_databases, which only creates the workbook-sourced tables),
    # llm_call_log/agent_execution_log don't exist yet, so
    # call_openai_structured's own record_llm_generation() write fails
    # silently and openai_patch.py's SQLite read-back always returns None.
    # Discovered during Task 13 manual verification.
    ensure_schema()

    run_id = payload.get("run_id") or str(uuid.uuid4())
    app = build_agent_graph(payload)

    node_latencies_ms: Dict[str, float] = {}
    llm_calls: list[LLMCallRecord] = []

    with patch_openai_calls(on_call=llm_calls.append):
        state = GlobalState(run_id=run_id)
        node_start = time.monotonic()
        for update in app.stream(GlobalState(run_id=run_id), stream_mode="updates"):
            for node_name, delta in update.items():
                elapsed_ms = (time.monotonic() - node_start) * 1000.0
                try:
                    node_latencies_ms[node_name] = elapsed_ms
                except Exception as exc:
                    logger.warning("Latency capture failed for %s (non-blocking): %s", node_name, exc)
                state = state.model_copy(update=delta)
                node_start = time.monotonic()

    labeled_latencies = {
        NODE_LATENCY_LABELS.get(name, name): ms for name, ms in node_latencies_ms.items()
    }
    labeled_latencies["total"] = sum(node_latencies_ms.values())

    _record_pipeline_run(
        run_id=run_id,
        node_latencies_ms=labeled_latencies,
        llm_calls=llm_calls,
        l4_signals=extract_l4_signals(state),
        l5_forecast=extract_l5_forecast(state),
    )

    return state
