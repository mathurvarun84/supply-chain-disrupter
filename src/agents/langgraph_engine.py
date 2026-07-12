import importlib.util
import logging
import uuid
from typing import Any, Callable, Dict

from langgraph.graph import END, START, StateGraph

from src.agents.data_ingestion.agent import data_ingestion_agent
from src.agents.mitigation_agent import mitigation_recommendation_agent
from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.risk_classifier_agent.agent import risk_classifier_agent
from src.agents.simulation_agent import simulation_agent
from src.agents.state import ForecastResult, GlobalState
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.utils.db_utils import fetch_time_series

logger = logging.getLogger(__name__)

# Optional heavy dependencies. Agents that need these degrade gracefully when absent.
_PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None
_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


# Bootstrap ingestion schema once per process (additive, never modifies lite_master).
try:
    from src.agents.data_ingestion_agent import data_ingestion_agent_v2
    from src.utils.ingestion_schema import ensure_ingestion_schema

    ensure_ingestion_schema()
    _INGESTION_V2_AVAILABLE = True
except Exception as _ingestion_bootstrap_exc:
    logger.warning("Ingestion schema bootstrap failed: %s", _ingestion_bootstrap_exc)
    _INGESTION_V2_AVAILABLE = False


def demand_forecasting_agent(state: GlobalState) -> Dict[str, Any]:
    """L5 - Prophet demand forecast (optional; skipped if prophet/pandas absent)."""
    if not _PROPHET_AVAILABLE or not _PANDAS_AVAILABLE:
        logger.warning("L5: prophet/pandas not installed; demand forecasting skipped.")
        return {
            "agent_logs": state.agent_logs + [
                "L5: SKIPPED - prophet or pandas not installed. Run: pip install prophet pandas"
            ],
        }

    if state.active_record is None:
        raise ValueError("Active record is required for demand forecasting.")

    ts = fetch_time_series(state.active_record["port"], state.active_record["sku"])
    if len(ts) < 10:
        return {
            "agent_logs": state.agent_logs + [
                f"L5: SKIPPED - only {len(ts)} history points available (need >= 10)."
            ],
        }

    import pandas as pd
    from prophet import Prophet

    df_records = [{"ds": row["event_date"], "y": row["demand"]} for row in ts]
    df = pd.DataFrame(df_records)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast_points = forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")
    demand_baseline = float(state.active_record.get("demand", 0.0))
    expected_drop = max(0.0, 1.0 - (forecast_points[-1]["yhat"] / (demand_baseline or 1.0)))
    return {
        "forecast_result": ForecastResult(
            prophet_forecast=forecast_points,
            expected_drop_pct=round(expected_drop * 100.0, 2),
        ),
        "agent_logs": state.agent_logs + ["L5: Demand forecasting completed."],
    }


def _merge_state(state: GlobalState, delta: Dict[str, Any]) -> GlobalState:
    """Apply an agent delta to GlobalState using the existing project contract."""
    return state.model_copy(update=delta)


def _run_optional(state: GlobalState, agent_fn, label: str) -> GlobalState:
    """Run an optional agent; on failure log SKIPPED and continue."""
    try:
        delta = agent_fn(state)
        return _merge_state(state, delta)
    except Exception as exc:
        logger.warning("%s skipped: %s", label, exc)
        return state.model_copy(
            update={"agent_logs": state.agent_logs + [f"{label}: SKIPPED - {exc}"]}
        )


def _l1_node(payload: Dict[str, Any]) -> Callable[[GlobalState], Dict[str, Any]]:
    """Create the L1 graph node because ingestion needs the external payload."""

    def _node(state: GlobalState) -> Dict[str, Any]:
        if _INGESTION_V2_AVAILABLE:
            try:
                return data_ingestion_agent_v2(state, payload)
            except Exception as _v2_exc:
                logger.warning("L1v2 failed, falling back to legacy: %s", _v2_exc)
                return data_ingestion_agent(state, payload)
        return data_ingestion_agent(state, payload)

    return _node


def _critical_node(
    agent_fn: Callable[[GlobalState], Dict[str, Any]],
) -> Callable[[GlobalState], Dict[str, Any]]:
    """Wrap critical-path agents so the graph receives a plain state delta."""

    def _node(state: GlobalState) -> Dict[str, Any]:
        return agent_fn(state)

    return _node


def _optional_node(
    agent_fn: Callable[[GlobalState], Dict[str, Any]],
    label: str,
) -> Callable[[GlobalState], Dict[str, Any]]:
    """Wrap optional agents with the existing skip-and-continue behavior."""

    def _node(state: GlobalState) -> Dict[str, Any]:
        try:
            return agent_fn(state)
        except Exception as exc:
            logger.warning("%s skipped: %s", label, exc)
            return {"agent_logs": state.agent_logs + [f"{label}: SKIPPED - {exc}"]}

    return _node


def build_agent_graph(payload: Dict[str, Any]):
    """
    Build the executable LangGraph pipeline for one scenario payload.

    Node order is intentionally conservative: L2 and L3 both depend only on L1,
    but they currently append to the same agent_logs list, so they stay
    sequential until the state model uses LangGraph reducers for parallel writes.
    """
    graph = StateGraph(GlobalState)

    graph.add_node("l1_data_ingestion", _l1_node(payload))
    graph.add_node("l2_news_analysis", _critical_node(news_event_analysis_agent))
    graph.add_node("l3_weather_monitoring", _critical_node(weather_risk_monitoring_agent))
    graph.add_node("l4_risk_classifier", _critical_node(risk_classifier_agent))
    graph.add_node("l5_demand_forecast", _optional_node(demand_forecasting_agent, "L5"))
    # L6 owns its Monte Carlo -> heuristic fallback.  Keep the orchestration
    # guard as a final safety net so L7 can still produce a plan without a
    # simulation result if an unexpected integration error escapes L6.
    graph.add_node("l6_simulation", _optional_node(simulation_agent, "L6"))
    # A mitigation plan is the terminal product of the pipeline, not an
    # optional enrichment.  Let failures surface to the caller instead of
    # returning an apparently successful state with mitigation_action=None.
    graph.add_node("l7_mitigation", _critical_node(mitigation_recommendation_agent))

    graph.add_edge(START, "l1_data_ingestion")
    graph.add_edge("l1_data_ingestion", "l2_news_analysis")
    graph.add_edge("l2_news_analysis", "l3_weather_monitoring")
    graph.add_edge("l3_weather_monitoring", "l4_risk_classifier")
    graph.add_edge("l4_risk_classifier", "l5_demand_forecast")
    graph.add_edge("l5_demand_forecast", "l6_simulation")
    graph.add_edge("l6_simulation", "l7_mitigation")
    graph.add_edge("l7_mitigation", END)

    return graph.compile()


def run_agent_graph(payload: Dict[str, Any]) -> GlobalState:
    """
    Execute the full LangGraph agent pipeline.

    Critical path: L1 -> L2 -> L3 -> L4 -> L7 (Mitigation)
    Optional enrichments before L7: L5 (Prophet) -> L6 (Simulation)
    """
    app = build_agent_graph(payload)
    result = app.invoke(GlobalState())
    if isinstance(result, GlobalState):
        return result
    return GlobalState.model_validate(result)


def run_agent_sequence(payload: Dict[str, Any]) -> GlobalState:
    """
    Sequential L1–L7 runner with full observability instrumentation.

    Each agent is wrapped by agent_span(), which writes agent_execution_log
    (unconditional) and a Langfuse span (best-effort). LLM agents additionally
    write llm_call_log rows via call_openai_structured()'s record_llm_generation
    hook. The pipeline_trace context flushes all Langfuse events at the end.

    run_id is taken from payload["run_id"] when present; a new UUID is minted
    otherwise so observability works even when called directly without one.
    """
    from src.utils.db_utils import ensure_schema, ensure_sku_id_columns
    from src.utils.observability import agent_span, pipeline_trace

    ensure_schema()
    ensure_sku_id_columns()

    run_id: str = payload.get("run_id") or str(uuid.uuid4())
    mode: str = payload.get("mode", "live")
    source_type: str = payload.get("source_type", "unknown")

    with pipeline_trace(run_id, mode=mode, source_type=source_type) as trace:
        state = GlobalState(run_id=run_id, langfuse_trace=trace)

        with agent_span(trace, run_id, "L1_ingestion") as span:
            state = state.model_copy(update={"langfuse_span": span})
            state = _merge_state(state, _l1_node(payload)(state))

        with agent_span(trace, run_id, "L2_news") as span:
            state = state.model_copy(update={"langfuse_span": span})
            state = _merge_state(state, news_event_analysis_agent(state))

        with agent_span(trace, run_id, "L3_weather") as span:
            state = state.model_copy(update={"langfuse_span": span})
            state = _merge_state(state, weather_risk_monitoring_agent(state))

        with agent_span(trace, run_id, "L4_risk_classifier") as span:
            state = state.model_copy(update={"langfuse_span": span})
            state = _merge_state(state, risk_classifier_agent(state))

        # L5/L6 are optional. Exceptions must propagate INTO agent_span so it
        # can write Failed-Fallback before re-raising; we catch the re-raise outside
        # the `with` block and swallow it with a SKIPPED log entry.
        try:
            with agent_span(trace, run_id, "L5_forecast") as span:
                state = state.model_copy(update={"langfuse_span": span})
                state = _merge_state(state, demand_forecasting_agent(state))
        except Exception as exc:
            logger.warning("L5 skipped: %s", exc)
            state = state.model_copy(
                update={"agent_logs": state.agent_logs + [f"L5: SKIPPED - {exc}"]}
            )

        try:
            with agent_span(trace, run_id, "L6_simulation") as span:
                state = state.model_copy(update={"langfuse_span": span})
                state = _merge_state(state, simulation_agent(state))
        except Exception as exc:
            logger.warning("L6 skipped: %s", exc)
            state = state.model_copy(
                update={"agent_logs": state.agent_logs + [f"L6: SKIPPED - {exc}"]}
            )

        # Match the compiled graph: L7 is required. The span records a failure,
        # and the exception then surfaces to the caller.
        with agent_span(trace, run_id, "L7_mitigation") as span:
            state = state.model_copy(update={"langfuse_span": span})
            state = _merge_state(state, mitigation_recommendation_agent(state))

    return state
