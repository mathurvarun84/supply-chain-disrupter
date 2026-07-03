"""
test_ragas_l4_live.py — Tests for the L4-specific RAGAS live-eval harness
(evaluation/ragas/run_l4_live_evaluation.py).

All tests run WITHOUT a real API key, WITHOUT a live ChromaDB, and WITHOUT a
real SQLite DB — risk_classifier_agent, retrieve_and_rerank,
call_openai_structured, execute_query, and ragas.evaluate are all mocked.

Run: python -m pytest tests/test_ragas_l4_live.py -v --tb=short
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pandas as pd
import pytest

import evaluation.ragas.run_l4_live_evaluation as run_l4


class _FakeLLMSignal:
    def __init__(self, rationale="grounded rationale", label="HIGH"):
        self.predicted_label = label
        self.rationale = rationale
        self.confidence_level = "high"
        self.primary_driver = "geo"
        self.rag_chunks_used = 1


def _patch_agent_pipeline(monkeypatch, fake_agent):
    monkeypatch.setattr("src.agents.risk_classifier_agent.risk_classifier_agent", fake_agent)
    monkeypatch.setattr("src.agents.risk_classifier_agent.agent.insert_risk_classification", lambda **k: None)
    monkeypatch.setattr("src.agents.risk_classifier_agent.agent.update_risk_label", lambda *a, **k: None)
    monkeypatch.setattr("src.agents.risk_classifier_agent.agent.run_judge", lambda **k: None)


def _record(order_id=1, delivery_status="Late delivery") -> dict:
    return {
        "order_id": order_id,
        "record_id": order_id,
        "delivery_status": delivery_status,
        "disruption_event_label": "HIGH",
        "port": "Shanghai",
    }


# ---------------------------------------------------------------------------
# 1. Signal3Capture patches and restores, even on exception
# ---------------------------------------------------------------------------

def test_signal3_capture_patches_and_restores():
    import src.agents.risk_classifier_agent.llm_signal as llm_signal_mod
    import src.rag.retriever as retriever_mod

    orig_retrieve = retriever_mod.retrieve_and_rerank
    orig_call_llm = llm_signal_mod.call_openai_structured

    with pytest.raises(ValueError, match="boom"):
        with run_l4.Signal3Capture():
            assert retriever_mod.retrieve_and_rerank is not orig_retrieve
            assert llm_signal_mod.call_openai_structured is not orig_call_llm
            raise ValueError("boom")

    assert retriever_mod.retrieve_and_rerank is orig_retrieve
    assert llm_signal_mod.call_openai_structured is orig_call_llm


# ---------------------------------------------------------------------------
# 2. Stratified sampling queries each bucket
# ---------------------------------------------------------------------------

def test_fetch_stratified_orders_queries_each_bucket(monkeypatch):
    calls = []

    def fake_execute_query(sql, params):
        calls.append(params)
        status = params[0]
        return [{"order_id": f"{status}-1"}]

    monkeypatch.setattr("src.utils.db_utils.execute_query", fake_execute_query)

    orders = run_l4.fetch_stratified_orders(n_per_bucket=2)

    assert len(calls) == len(run_l4.DELIVERY_STATUS_BUCKETS)
    assert {c[0] for c in calls} == set(run_l4.DELIVERY_STATUS_BUCKETS)
    assert all(c[1] == 2 for c in calls)
    assert len(orders) == len(run_l4.DELIVERY_STATUS_BUCKETS)


# ---------------------------------------------------------------------------
# 3. Successful capture — question/contexts/answer populated
# ---------------------------------------------------------------------------

def test_run_order_and_capture_success(monkeypatch):
    import src.agents.risk_classifier_agent.llm_signal as llm_signal_mod
    import src.rag.retriever as retriever_mod

    fake_hits = [{"text": "grounded chunk text", "metadata": {}}]
    monkeypatch.setattr(retriever_mod, "retrieve_and_rerank", lambda *a, **k: fake_hits)
    monkeypatch.setattr(llm_signal_mod, "call_openai_structured", lambda *a, **k: _FakeLLMSignal())

    def fake_agent(state):
        retriever_mod.retrieve_and_rerank(query="disruption query", collections=["historical_precedents"])
        llm_signal_mod.call_openai_structured("sys", "user", response_model=object, model="gpt-4o")
        return {}

    _patch_agent_pipeline(monkeypatch, fake_agent)

    result = run_l4.run_order_and_capture(_record())

    assert result["evaluated"] is True
    assert result["question"] == "disruption query"
    assert result["contexts"] == ["grounded chunk text"]
    assert result["answer"] == "grounded rationale"
    assert result["predicted_label"] == "HIGH"


# ---------------------------------------------------------------------------
# 4. LLMSignal is None (e.g. no API key) — skipped, not scored
# ---------------------------------------------------------------------------

def test_run_order_and_capture_skips_when_llm_signal_none(monkeypatch):
    def fake_agent(state):
        return {}  # never calls retrieve_and_rerank or call_openai_structured

    _patch_agent_pipeline(monkeypatch, fake_agent)

    result = run_l4.run_order_and_capture(_record())

    assert result["evaluated"] is False
    assert result["skipped_reason"] == "llm_signal_none"


# ---------------------------------------------------------------------------
# 5. LLMSignal present but no retrieval happened — skipped, not scored
# ---------------------------------------------------------------------------

def test_run_order_and_capture_skips_when_no_context(monkeypatch):
    import src.agents.risk_classifier_agent.llm_signal as llm_signal_mod

    monkeypatch.setattr(llm_signal_mod, "call_openai_structured", lambda *a, **k: _FakeLLMSignal())

    def fake_agent(state):
        llm_signal_mod.call_openai_structured("sys", "user", response_model=object, model="gpt-4o")
        return {}

    _patch_agent_pipeline(monkeypatch, fake_agent)

    result = run_l4.run_order_and_capture(_record())

    assert result["evaluated"] is False
    assert result["skipped_reason"] == "no_context_retrieved"
    assert result["answer"] == "grounded rationale"  # rationale still recorded even though excluded from RAGAS


# ---------------------------------------------------------------------------
# 6. Agent exception is caught, never crashes the batch
# ---------------------------------------------------------------------------

def test_run_order_and_capture_handles_agent_exception(monkeypatch):
    def fake_agent(state):
        raise RuntimeError("db locked")

    _patch_agent_pipeline(monkeypatch, fake_agent)

    result = run_l4.run_order_and_capture(_record())

    assert result["evaluated"] is False
    assert "agent_call_failed" in result["skipped_reason"]


# ---------------------------------------------------------------------------
# 7. Cost guard blocks without --yes
# ---------------------------------------------------------------------------

def test_cost_guard_blocks_without_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    assert run_l4.cost_guard(25, skip_confirmation=False) is False


def test_cost_guard_skips_prompt_with_yes(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr("builtins.input", called)
    assert run_l4.cost_guard(25, skip_confirmation=True) is True
    called.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Flagging below target
# ---------------------------------------------------------------------------

def test_flag_weak_below_target():
    by_bucket = {
        "Shipping canceled": {"faithfulness": 0.5, "answer_relevancy": 0.9, "n_cases": 2},
        "Late delivery": {"faithfulness": 0.9, "answer_relevancy": 0.9, "n_cases": 2},
    }
    flagged = run_l4.flag_weak(by_bucket)

    assert len(flagged) == 1
    assert flagged[0]["delivery_status"] == "Shipping canceled"
    assert flagged[0]["metric"] == "faithfulness"
    assert flagged[0]["gap"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# 9. main() exits fast without OPENAI_API_KEY — never touches the DB
# ---------------------------------------------------------------------------

def test_main_exits_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["run_l4_live_evaluation.py", "--n-per-bucket", "1"])

    db_touch = MagicMock()
    monkeypatch.setattr(run_l4, "fetch_stratified_orders", db_touch)

    exit_code = run_l4.main()

    assert exit_code == 1
    db_touch.assert_not_called()


# ---------------------------------------------------------------------------
# 10. run_ragas_l4_metrics builds a ground_truth-free Dataset
# ---------------------------------------------------------------------------

def test_run_ragas_l4_metrics_no_ground_truth_column(monkeypatch):
    captured = {}

    class FakeDataset:
        @staticmethod
        def from_dict(d):
            captured["dict"] = d
            return d

    monkeypatch.setattr("datasets.Dataset", FakeDataset)
    monkeypatch.setattr(run_l4, "_build_ragas_llm_and_embeddings", lambda: (MagicMock(), MagicMock()))

    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame(
        [{"faithfulness": 0.9, "answer_relevancy": 0.9}]
    )
    monkeypatch.setattr("ragas.evaluate", lambda *a, **k: fake_result)

    records = [{"question": "q", "contexts": ["c"], "answer": "a"}]
    df = run_l4.run_ragas_l4_metrics(records)

    assert "ground_truth" not in captured["dict"]
    assert set(captured["dict"].keys()) == {"question", "contexts", "answer"}
    assert df.iloc[0]["faithfulness"] == 0.9
