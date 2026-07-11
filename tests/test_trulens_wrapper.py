import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import GlobalState
from src.evaluation.trulens_integration.wrapper import run_with_trulens


class _FakeCompiledGraph:
    """Stands in for build_agent_graph(payload)'s return value."""

    def stream(self, initial_state, stream_mode):
        assert stream_mode == "updates"
        assert isinstance(initial_state, GlobalState)
        assert initial_state.run_id is not None
        yield {"l1_data_ingestion": {"agent_logs": ["L1: ok"]}}
        yield {"l2_news_analysis": {"agent_logs": ["L1: ok", "L2: ok"]}}
        yield {"l7_mitigation": {"agent_logs": ["L1: ok", "L2: ok", "L7: ok"]}}


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


def test_run_with_trulens_records_per_node_latency():
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        with patch(
            "src.evaluation.trulens_integration.wrapper._record_pipeline_run"
        ) as mock_record:
            run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert mock_record.called
    _, kwargs = mock_record.call_args
    latencies = kwargs["node_latencies_ms"]
    # wrapper.py relabels raw node names (l1_data_ingestion, ...) via
    # NODE_LATENCY_LABELS to the short form node_latency_check() expects
    # (L1, L2, ...), and adds a "total" entry summing all node latencies.
    assert set(latencies.keys()) == {"L1", "L2", "L7", "total"}
    assert all(v >= 0.0 for v in latencies.values())
    assert latencies["total"] == pytest.approx(
        latencies["L1"] + latencies["L2"] + latencies["L7"]
    )


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
