import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import GlobalState
from src.evaluation.trulens_integration.wrapper import _pipeline_body, run_with_trulens


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
