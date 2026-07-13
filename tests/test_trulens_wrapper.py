import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import GlobalState
from src.evaluation.trulens_integration.openai_patch import LLMCallRecord
from src.evaluation.trulens_integration.wrapper import (
    _aggregate_llm_cost, _pipeline_body, run_with_trulens,
)


class _FakeCompiledGraph:
    """Stands in for build_agent_graph(payload)'s return value."""

    def stream(self, initial_state, stream_mode):
        assert stream_mode == "updates"
        assert isinstance(initial_state, GlobalState)
        assert initial_state.run_id is not None
        yield {"l1_data_ingestion": {"agent_logs": ["L1: ok"]}}
        yield {"l2_news_analysis": {"agent_logs": ["L1: ok", "L2: ok"]}}
        yield {"l7_mitigation": {"agent_logs": ["L1: ok", "L2: ok", "L7: ok"]}}


def test_run_with_trulens_ensures_schema_before_running():
    # llm_call_log/agent_execution_log must exist before the graph runs, or
    # call_openai_structured's own record_llm_generation() write fails
    # silently and openai_patch.py's token read-back always returns None.
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        with patch("src.utils.db_utils.ensure_schema") as mock_ensure_schema:
            run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    mock_ensure_schema.assert_called_once()


def test_run_with_trulens_returns_merged_final_state():
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        result = run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert isinstance(result, GlobalState)
    assert result.agent_logs == ["L1: ok", "L2: ok", "L7: ok"]
    assert result.run_id is not None


def test_run_with_trulens_mints_run_id_when_absent_from_payload():
    captured_state = {}

    class _CapturingGraph(_FakeCompiledGraph):
        def stream(self, initial_state, stream_mode):
            captured_state["run_id"] = initial_state.run_id
            return super().stream(initial_state, stream_mode)

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_CapturingGraph(),
    ):
        run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert captured_state["run_id"]


def test_run_with_trulens_uses_run_id_from_payload_when_present():
    class _CapturingGraph(_FakeCompiledGraph):
        captured = {}

        def stream(self, initial_state, stream_mode):
            self.captured["run_id"] = initial_state.run_id
            return super().stream(initial_state, stream_mode)

    graph = _CapturingGraph()
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=graph,
    ):
        run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03", "run_id": "fixed-id-123"})

    assert graph.captured["run_id"] == "fixed-id-123"


def test_pipeline_body_labels_per_node_latency():
    # _pipeline_body is the pure core logic (no TruLens dependency) —
    # unit-tested directly rather than through the real TruApp recording
    # path that run_with_trulens() now wraps it in.
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        state, latencies, llm_calls = _pipeline_body(
            {"sku": "CHIP_AP", "event_date": "2024-04-03"}, run_id="test-run-id"
        )

    assert isinstance(state, GlobalState)
    assert llm_calls == []
    # wrapper.py relabels raw node names (l1_data_ingestion, ...) via
    # NODE_LATENCY_LABELS to the short form node_latency_check() expects
    # (L1, L2, ...), and adds a "total" entry summing all node latencies.
    assert set(latencies.keys()) == {"L1", "L2", "L7", "total"}
    assert all(v >= 0.0 for v in latencies.values())
    assert latencies["total"] == pytest.approx(
        latencies["L1"] + latencies["L2"] + latencies["L7"]
    )


def test_run_with_trulens_records_a_real_trulens_event():
    # Unlike the other tests here, this deliberately does NOT mock TruLens —
    # it's the one test proving run_with_trulens() actually writes something
    # TruLens's own dashboard can read, not just that our own telemetry
    # capture (_pipeline_body) runs correctly.
    import sqlite3

    from src.evaluation.trulens_integration.config import DB_PATH, get_session

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03", "run_id": "trulens-event-test"})

    get_session().force_flush()
    conn = sqlite3.connect(DB_PATH)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM trulens_events WHERE record_attributes LIKE '%supply_chain_pipeline%'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count > 0


def _llm_call(input_tokens, output_tokens, cost_usd, model="gpt-4o"):
    return LLMCallRecord(
        run_id="r", agent_name="L2_news", model=model, system_prompt="s", user_message="u",
        latency_ms=100.0, input_tokens=input_tokens, output_tokens=output_tokens,
        cost_usd=cost_usd, status="success", parsed_output=None,
    )


def test_aggregate_llm_cost_sums_across_calls():
    calls = [_llm_call(100, 50, 0.001), _llm_call(200, 80, 0.002, model="gpt-4.1-mini")]
    summary = _aggregate_llm_cost(calls)
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 130
    assert summary["cost_usd"] == pytest.approx(0.003)
    assert summary["models"] == ["gpt-4.1-mini", "gpt-4o"]


def test_aggregate_llm_cost_handles_empty_list():
    summary = _aggregate_llm_cost([])
    assert summary == {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "models": []}


def test_aggregate_llm_cost_treats_none_fields_as_zero():
    # openai_patch.py's LLMCallRecord leaves tokens/cost as None on a failed call
    calls = [_llm_call(None, None, None)]
    summary = _aggregate_llm_cost(calls)
    assert summary == {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "models": ["gpt-4o"]}


def test_run_with_trulens_calls_get_session_before_constructing_tru_app():
    # Regression test: TruApp() constructed before get_session() silently
    # creates its own default TruSession pointed at sqlite:///default.sqlite
    # in the CWD instead of data/trulens/trulens.db, with no error anywhere
    # in the chain — get_session() must be called first. get_session() is
    # @lru_cache'd, so a plain "was it called" assertion would pass whether
    # or not call order was fixed (an earlier test may have already warmed
    # the cache) — this asserts the actual order instead.
    call_order = []

    def fake_get_session():
        call_order.append("get_session")
        return MagicMock()

    def fake_tru_app(*args, **kwargs):
        call_order.append("TruApp")
        return MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        with patch("src.evaluation.trulens_integration.config.get_session", fake_get_session):
            with patch("src.evaluation.trulens_integration.wrapper.TruApp", fake_tru_app):
                run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert call_order == ["get_session", "TruApp"]


def test_run_with_trulens_records_cost_and_token_span_attributes():
    import json
    import sqlite3

    from src.evaluation.trulens_integration.config import DB_PATH, get_session

    fake_calls = [_llm_call(400, 100, 0.0042, model="gpt-4o")]

    with patch(
        "src.evaluation.trulens_integration.wrapper._pipeline_body",
        return_value=(GlobalState(risk_classification=None), {"L2": 1000.0, "total": 1000.0}, fake_calls),
    ):
        run_with_trulens({
            "affected_port": "Chennai", "sku": "CHIP_AP", "event_date": "2024-04-03",
            "run_id": "cost-attr-test-run",
        })

    get_session().force_flush()
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT record_attributes FROM trulens_events "
            "WHERE record_attributes LIKE '%cost-attr-test-run%' "
            "AND record_attributes LIKE '%record_root%' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    attrs = json.loads(row[0])
    assert attrs["ai.observability.cost.cost"] == pytest.approx(0.0042)
    assert attrs["ai.observability.cost.cost_currency"] == "USD"
    assert attrs["ai.observability.cost.num_prompt_tokens"] == 400
    assert attrs["ai.observability.cost.num_completion_tokens"] == 100


def test_run_with_trulens_lets_node_exceptions_propagate():
    class _RaisingGraph:
        def stream(self, initial_state, stream_mode):
            yield {"l1_data_ingestion": {"agent_logs": ["L1: ok"]}}
            raise ValueError("Risk label is required for mitigation")

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_RaisingGraph(),
    ):
        try:
            run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})
            assert False, "expected ValueError to propagate"
        except ValueError as exc:
            assert "Risk label is required" in str(exc)
