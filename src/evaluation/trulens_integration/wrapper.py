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

Recording into TruLens's own dashboard uses TruApp + @instrument (the
current, OTEL-native API — verified against the installed trulens-core
2.8.1: the legacy TruVirtual.add_record() path raises "Not supported with
OTel tracing enabled!" since OTEL tracing is this version's default).
_PipelineRunner.run() is the @instrument-decorated span; wrapping it in
`with tru_app as recording:` still lets real exceptions propagate
unchanged (verified directly against the installed package) — only
session.force_flush() afterward is best-effort/non-blocking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from opentelemetry import trace as ot_trace
from trulens.apps.app import TruApp, instrument
from trulens.core.otel.instrument import set_user_defined_attributes
from trulens.otel.semconv.trace import SpanAttributes

from src.agents.langgraph_engine import build_agent_graph
from src.agents.state import GlobalState
from src.evaluation.trulens_integration.node_extractors import (
    NODE_LATENCY_LABELS, extract_l4_signals, extract_l5_forecast,
)
from src.evaluation.trulens_integration.openai_patch import LLMCallRecord, patch_openai_calls

logger = logging.getLogger(__name__)


def _pipeline_body(payload: Dict[str, Any], run_id: str) -> tuple[GlobalState, Dict[str, float], list]:
    """Core pipeline execution: stream the graph, patch OpenAI calls, merge state.

    Pure with respect to TruLens — no session/recording concerns here, so
    this is testable without a real TruLens session. Returns
    (final_state, labeled_node_latencies_ms, llm_call_records).
    """
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

    return state, labeled_latencies, llm_calls


def _aggregate_llm_cost(llm_calls: list[LLMCallRecord]) -> Dict[str, Any]:
    """Sum tokens/cost across one run's LLM calls for TruLens's SpanAttributes.COST.*.

    None fields (failed calls in openai_patch.py leave tokens/cost as None)
    count as zero rather than breaking the sum.
    """
    return {
        "prompt_tokens": sum(c.input_tokens or 0 for c in llm_calls),
        "completion_tokens": sum(c.output_tokens or 0 for c in llm_calls),
        "cost_usd": round(sum(c.cost_usd or 0.0 for c in llm_calls), 6),
        "models": sorted({c.model for c in llm_calls if c.model}),
    }


class _PipelineRunner:
    """Holds one run's result so the @instrument-decorated span has a plain
    method to wrap. TruApp auto-captures run()'s args/return value as the
    span's root input/output, but only for simple (string) types — a dict
    return value shows up as an empty output in the dashboard, and any
    manual override of record_root.input/output set mid-call gets
    overwritten by @instrument's own post-call finalization (both verified
    against the installed trulens-core 2.8.1). So run() returns a plain
    risk-label string for a readable "Output" column, and cost/token/scenario
    detail is attached via set_user_defined_attributes under distinct keys
    that @instrument doesn't touch."""

    def __init__(self) -> None:
        self.final_state: GlobalState | None = None
        self.node_latencies_ms: Dict[str, float] = {}
        self.llm_calls: list[LLMCallRecord] = []

    @instrument
    def run(self, payload: Dict[str, Any], run_id: str) -> str:
        state, node_latencies_ms, llm_calls = _pipeline_body(payload, run_id)
        self.final_state = state
        self.node_latencies_ms = node_latencies_ms
        self.llm_calls = llm_calls

        cost_summary = _aggregate_llm_cost(llm_calls)
        try:
            span = ot_trace.get_current_span()
            set_user_defined_attributes(span, attributes={
                SpanAttributes.COST.COST: cost_summary["cost_usd"],
                SpanAttributes.COST.CURRENCY: "USD",
                SpanAttributes.COST.NUM_PROMPT_TOKENS: cost_summary["prompt_tokens"],
                SpanAttributes.COST.NUM_COMPLETION_TOKENS: cost_summary["completion_tokens"],
                "supply_chain.scenario": (
                    f"port={payload.get('affected_port', '?')} "
                    f"sku={payload.get('sku', '?')} event_date={payload.get('event_date', '?')}"
                ),
                "supply_chain.node_latencies_ms": json.dumps(node_latencies_ms),
                "supply_chain.l4_signals": json.dumps(extract_l4_signals(state)),
                "supply_chain.l5_forecast": json.dumps(extract_l5_forecast(state)),
            })
        except Exception as exc:
            logger.warning("TruLens span attribute capture failed (non-blocking) for run_id=%s: %s", run_id, exc)

        return state.risk_label or "UNKNOWN"


def run_with_trulens(payload: Dict[str, Any], capture: Optional[Dict[str, Any]] = None) -> GlobalState:
    """Drop-in replacement for run_agent_graph(payload) with TruLens tracing.

    `capture`, when passed, is filled in-place with this run's
    node_latencies_ms and cost_summary (the same values recorded onto the
    TruLens span — see _PipelineRunner.run()) before returning, so a caller
    that needs them (the TruLens tab's capture-run endpoint,
    src/api/routers/trulens.py) doesn't have to re-query TruLens's own
    OTEL-backed SQLite schema for data this process already computed.
    Purely additive — omitting `capture` preserves the exact prior signature
    every existing caller (cli.py, evaluation/trulens_runner.py) relies on.
    """
    from src.utils.db_utils import ensure_schema

    # Neither run_agent_graph() nor build_agent_graph() calls ensure_schema()
    # — on a freshly-built outputs/supply_chain.db (rebuilt via
    # src.build_databases, which only creates the workbook-sourced tables),
    # llm_call_log/agent_execution_log don't exist yet, so
    # call_openai_structured's own record_llm_generation() write fails
    # silently and openai_patch.py's SQLite read-back always returns None.
    # Discovered during Task 13 manual verification.
    ensure_schema()

    # get_session() MUST be called before TruApp() is constructed. TruLens's
    # TruSession is a process-wide singleton registered lazily — if no
    # session has been created yet, TruApp() silently creates its own
    # default one pointed at sqlite:///default.sqlite in the CWD instead of
    # data/trulens/trulens.db, and every span records successfully into the
    # wrong file with no error anywhere in the chain. Discovered by
    # comparing the exact same attribute-setting code (which worked in an
    # isolated smoke test that called get_session() first) against
    # run_with_trulens() silently writing to ./default.sqlite instead.
    from src.evaluation.trulens_integration.config import get_session

    session = get_session()

    run_id = payload.get("run_id") or str(uuid.uuid4())
    runner = _PipelineRunner()
    tru_app = TruApp(runner, app_name="supply_chain_pipeline", app_version="v1", main_method=runner.run)

    with tru_app as recording:
        runner.run(payload, run_id)

    try:
        session.force_flush()
    except Exception as exc:
        logger.warning("TruLens flush failed (non-blocking) for run_id=%s: %s", run_id, exc)

    assert runner.final_state is not None
    if capture is not None:
        capture["node_latencies_ms"] = runner.node_latencies_ms
        capture["cost_summary"] = _aggregate_llm_cost(runner.llm_calls)
    return runner.final_state
