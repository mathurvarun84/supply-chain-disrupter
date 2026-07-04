"""
test_ragas_phase3.py — Tests for RAGAS Phase 3 (evaluation runner).

All tests run WITHOUT a real API key and WITHOUT a live ChromaDB — every
external dependency (retrieve_and_rerank, call_openai_structured,
get_embedding_model, ragas.evaluate) is mocked.

Run: python -m pytest tests/test_ragas_phase3.py -v --tb=short
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pandas as pd
import pytest

import evaluation.ragas.run_evaluation as run_eval


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        mode="retrieval-only",
        limit=None,
        collections=None,
        styles=None,
        yes=False,
        rerank_top_k=3,
        bi_encoder_top_n=10,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _case(question: str, collection: str, chunk_id: str = "gold1", style: str = "natural_question") -> dict:
    return {
        "question": question,
        "ground_truth": f"ground truth for {question}",
        "source_collection": collection,
        "source_chunk_id": chunk_id,
        "query_style": style,
    }


# ---------------------------------------------------------------------------
# 1. Hit rate / MRR computation
# ---------------------------------------------------------------------------

def test_hit_rate_and_mrr_computation():
    records = [
        {"hit": True, "rank": 1, "context_relevance": 0.5, "context_recall_proxy": 0.5},
        {"hit": True, "rank": 2, "context_relevance": 0.5, "context_recall_proxy": 0.5},
        {"hit": False, "rank": None, "context_relevance": 0.0, "context_recall_proxy": 0.0},
        {"hit": False, "rank": None, "context_relevance": 0.0, "context_recall_proxy": 0.0},
    ]
    agg = run_eval._aggregate_group(records)
    assert agg["hit_rate_at_k"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)
    assert agg["n_cases"] == 4


# ---------------------------------------------------------------------------
# 2. Context relevance uses the shared embedding model
# ---------------------------------------------------------------------------

def test_context_relevance_uses_shared_embedding_model(monkeypatch, tmp_path):
    calls = []

    def fake_embedding_fn(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    fake_get_embedding_model = MagicMock(return_value=fake_embedding_fn)
    monkeypatch.setattr("src.utils.rag_utils.get_embedding_model", fake_get_embedding_model)
    monkeypatch.setattr(run_eval, "run_retrieval", lambda *a, **k: [{"text": "some chunk text", "metadata": {}}])

    cases = [_case("What is X?", "export_control_corpus")]
    monkeypatch.setattr(run_eval, "RETRIEVAL_ONLY_OUTPUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path / "traces")

    run_eval.run_retrieval_only_mode(cases, "chromadb_native", _args())

    fake_get_embedding_model.assert_called_once()
    assert calls, "embedding function was never invoked"


# ---------------------------------------------------------------------------
# 3. Retrieval-only mode runs without an API key
# ---------------------------------------------------------------------------

def test_retrieval_only_runs_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_embedding_fn(texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.utils.rag_utils.get_embedding_model", lambda: fake_embedding_fn)
    monkeypatch.setattr(
        run_eval, "run_retrieval", lambda *a, **k: [{"text": "chunk text", "metadata": {}}]
    )

    output_path = tmp_path / "ragas_scores_retrieval_only.json"
    monkeypatch.setattr(run_eval, "RETRIEVAL_ONLY_OUTPUT_PATH", output_path)
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path / "traces")

    cases = [_case("What is export control?", "export_control_corpus")]
    exit_code = run_eval.run_retrieval_only_mode(cases, "chromadb_native", _args())

    assert exit_code == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 4. Full mode exits (does not crash) without an API key
# ---------------------------------------------------------------------------

def test_full_mode_exits_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    output_path = tmp_path / "ragas_scores_full.json"
    monkeypatch.setattr(run_eval, "FULL_OUTPUT_PATH", output_path)

    cases = [_case("What is export control?", "export_control_corpus")]
    exit_code = run_eval.run_full_mode(cases, _args(mode="full"))

    assert exit_code == 1
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 5. Cost guard blocks without --yes
# ---------------------------------------------------------------------------

def test_cost_guard_blocks_without_yes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")

    generation_calls = MagicMock()
    monkeypatch.setattr("src.utils.openai_utils.call_openai_structured", generation_calls)

    output_path = tmp_path / "ragas_scores_full.json"
    monkeypatch.setattr(run_eval, "FULL_OUTPUT_PATH", output_path)

    cases = [_case(f"Question {i}", "export_control_corpus") for i in range(25)]
    exit_code = run_eval.run_full_mode(cases, _args(mode="full", yes=False))

    assert exit_code == 1
    generation_calls.assert_not_called()
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 6. Zero-context case is excluded from the ragas Dataset
# ---------------------------------------------------------------------------

def test_zero_context_case_excluded_from_ragas_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    no_context_case = _case("No retrievable question", "export_control_corpus", chunk_id="gold_a")
    has_context_case = _case("Answerable question", "india_sourcing_corpus", chunk_id="gold_b")

    def fake_run_retrieval(question, collection, top_n, top_k):
        if question == no_context_case["question"]:
            return []
        return [{"text": "grounded chunk text", "metadata": {"source_file": "doc.txt"}}]

    monkeypatch.setattr(run_eval, "run_retrieval", fake_run_retrieval)

    fake_answer = run_eval.RAGASAnswerOutput(answer="This is the grounded answer.")
    monkeypatch.setattr("src.utils.openai_utils.call_openai_structured", lambda *a, **k: fake_answer)

    captured = {}

    def fake_run_ragas_evaluate(evaluated_records):
        captured["records"] = evaluated_records
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                }
            ]
        )

    monkeypatch.setattr(run_eval, "run_ragas_evaluate", fake_run_ragas_evaluate)

    output_path = tmp_path / "ragas_scores_full.json"
    monkeypatch.setattr(run_eval, "FULL_OUTPUT_PATH", output_path)
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path / "traces")

    cases = [no_context_case, has_context_case]
    exit_code = run_eval.run_full_mode(cases, _args(mode="full", yes=True))

    assert exit_code == 0
    assert len(captured["records"]) == 1
    assert captured["records"][0]["question"] == has_context_case["question"]

    import json

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    per_case_by_q = {r["question"]: r for r in payload["per_case"]}
    assert per_case_by_q[no_context_case["question"]]["evaluated"] is False
    assert per_case_by_q[no_context_case["question"]]["skipped_reason"] == "no_context_retrieved"
    assert per_case_by_q[has_context_case["question"]]["evaluated"] is True


# ---------------------------------------------------------------------------
# 7. Flagging below target
# ---------------------------------------------------------------------------

def test_flagging_below_target():
    by_collection = {
        "india_sourcing_corpus": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "context_precision": 0.68,
            "context_recall": 0.9,
            "n_cases": 3,
        },
        "export_control_corpus": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "context_precision": 0.9,
            "context_recall": 0.9,
            "n_cases": 3,
        },
    }
    flagged = run_eval.flag_weak_collections(by_collection)

    assert len(flagged) == 1
    entry = flagged[0]
    assert entry["source_collection"] == "india_sourcing_corpus"
    assert entry["metric"] == "context_precision"
    assert entry["score"] == pytest.approx(0.68)
    assert entry["target"] == pytest.approx(0.75)
    assert entry["gap"] == pytest.approx(0.07)
    assert all(f["source_collection"] != "export_control_corpus" for f in flagged)


# ---------------------------------------------------------------------------
# 8. Filters are applied before evaluation
# ---------------------------------------------------------------------------

def test_filters_applied_before_evaluation(monkeypatch, tmp_path):
    cases = [
        _case("EC question 1", "export_control_corpus"),
        _case("EC question 2", "export_control_corpus"),
        _case("EC question 3", "export_control_corpus"),
        _case("IS question 1", "india_sourcing_corpus"),
        _case("HP question 1", "historical_precedents"),
    ]
    filtered = run_eval.apply_filters(cases, collections="export_control_corpus", styles=None, limit=2)
    assert len(filtered) == 2
    assert all(c["source_collection"] == "export_control_corpus" for c in filtered)

    seen_calls = []

    def fake_run_retrieval(question, collection, top_n, top_k):
        seen_calls.append((question, collection))
        return []

    def fake_embedding_fn(texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(run_eval, "run_retrieval", fake_run_retrieval)
    monkeypatch.setattr("src.utils.rag_utils.get_embedding_model", lambda: fake_embedding_fn)
    monkeypatch.setattr(run_eval, "RETRIEVAL_ONLY_OUTPUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr(run_eval, "TRACES_DIR", tmp_path / "traces")

    run_eval.run_retrieval_only_mode(filtered, "chromadb_native", _args())

    assert len(seen_calls) == 2
    assert all(collection == "export_control_corpus" for _, collection in seen_calls)
