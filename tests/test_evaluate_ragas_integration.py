"""
test_evaluate_ragas_integration.py — Tests for evaluate_all.py Phase 4 (RAGAS wiring).

All tests mock the filesystem only — no real ChromaDB, no API key, no ragas
import needed.

Run: python -m pytest tests/test_evaluate_ragas_integration.py -v --tb=short
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fine_tuning import evaluate_all


RETRIEVAL_ONLY_FIXTURE = {
    "mode": "retrieval-only",
    "run_at_utc": "2026-07-03T07:23:36.208519+00:00",
    "params": {"bi_encoder_top_n": 10, "rerank_top_k": 3},
    "overall": {
        "hit_rate_at_k": 1.0,
        "mrr": 0.9,
        "mean_context_relevance": 0.5937,
        "mean_context_recall_proxy": 0.8253,
        "n_cases": 5,
    },
    "by_collection": {
        "historical_precedents": {
            "hit_rate_at_k": 1.0,
            "mrr": 0.9,
            "mean_context_relevance": 0.5937,
            "mean_context_recall_proxy": 0.8253,
            "n_cases": 5,
        }
    },
    "by_style": {},
    "per_case": [],
}

FULL_FIXTURE = {
    "mode": "full",
    "run_at_utc": "2026-07-03T10:18:14.341249+00:00",
    "ragas_version": "0.4.3",
    "judge_model": "gpt-4.1-mini",
    "judge_embeddings": "text-embedding-3-small",
    "params": {"bi_encoder_top_n": 10, "rerank_top_k": 3},
    "n_cases_total": 6,
    "n_cases_evaluated": 6,
    "n_cases_skipped_no_context": 0,
    "overall": {
        "faithfulness": 1.0,
        "answer_relevancy": 0.6861,
        "context_precision": 0.9722,
        "context_recall": 1.0,
        "n_cases": 6,
    },
    "by_collection": {
        "historical_precedents": {
            "faithfulness": 1.0,
            "answer_relevancy": 0.6861,
            "context_precision": 0.9722,
            "context_recall": 1.0,
            "n_cases": 6,
        }
    },
    "by_style": {},
    "flagged": [
        {
            "source_collection": "historical_precedents",
            "metric": "answer_relevancy",
            "score": 0.6861,
            "target": 0.8,
            "gap": 0.1139,
        }
    ],
    "per_case": [],
}


def _write(tmp_path, name, data):
    p = tmp_path / name
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_both_files_missing_returns_none(tmp_path, caplog):
    retrieval_path = tmp_path / "ragas_scores_retrieval_only.json"
    full_path = tmp_path / "ragas_scores_full.json"
    with patch.object(Path, "exists", return_value=False):
        with caplog.at_level("WARNING"):
            result = evaluate_all.evaluate_ragas()
    assert result is None
    assert any("No RAGAS scores found" in r.message for r in caplog.records)


def test_retrieval_only_present_full_absent(tmp_path, caplog):
    retrieval_path = _write(tmp_path, "ragas_scores_retrieval_only.json", RETRIEVAL_ONLY_FIXTURE)
    full_path = tmp_path / "ragas_scores_full.json"

    with patch("fine_tuning.evaluate_all.Path") as MockPath:
        def side_effect(arg):
            if "retrieval_only" in arg:
                return retrieval_path
            if "full" in arg:
                return full_path
            return Path(arg)
        MockPath.side_effect = side_effect
        with caplog.at_level("WARNING"):
            result = evaluate_all.evaluate_ragas()

    assert result is not None
    assert result["retrieval_only"]["hit_rate_at_k"] == 1.0
    assert result["retrieval_only"]["mrr"] == 0.9
    assert result["retrieval_only"]["n_cases"] == 5
    assert result["full"] is None
    assert any("ragas_scores_full.json not found" in r.message for r in caplog.records)
    assert not any(r.levelname == "ERROR" for r in caplog.records)


def test_both_present(tmp_path):
    retrieval_path = _write(tmp_path, "ragas_scores_retrieval_only.json", RETRIEVAL_ONLY_FIXTURE)
    full_path = _write(tmp_path, "ragas_scores_full.json", FULL_FIXTURE)

    with patch("fine_tuning.evaluate_all.Path") as MockPath:
        def side_effect(arg):
            if "retrieval_only" in arg:
                return retrieval_path
            if "full" in arg:
                return full_path
            return Path(arg)
        MockPath.side_effect = side_effect
        result = evaluate_all.evaluate_ragas()

    assert result["retrieval_only"] is not None
    assert result["full"] is not None
    assert result["full"]["faithfulness"] == 1.0
    assert result["full"]["flagged"] == FULL_FIXTURE["flagged"]


def test_flagged_empty_list_no_warning(tmp_path, caplog):
    retrieval_path = _write(tmp_path, "ragas_scores_retrieval_only.json", RETRIEVAL_ONLY_FIXTURE)
    full_fixture_no_flag = dict(FULL_FIXTURE)
    full_fixture_no_flag["flagged"] = []
    full_path = _write(tmp_path, "ragas_scores_full.json", full_fixture_no_flag)

    with patch("fine_tuning.evaluate_all.Path") as MockPath:
        def side_effect(arg):
            if "retrieval_only" in arg:
                return retrieval_path
            if "full" in arg:
                return full_path
            return Path(arg)
        MockPath.side_effect = side_effect
        with caplog.at_level("WARNING"):
            result = evaluate_all.evaluate_ragas()

    assert result["full"]["flagged"] == []
    assert not any("weak" in r.message for r in caplog.records)


def test_run_all_evaluations_includes_ragas_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fine_tuning").mkdir()
    with patch.object(evaluate_all, "evaluate_distilbert", return_value={}), \
         patch.object(evaluate_all, "evaluate_embeddings", return_value={}), \
         patch.object(evaluate_all, "evaluate_cross_encoder_reranking", return_value={}), \
         patch.object(evaluate_all, "evaluate_gpt_finetuned", return_value={}), \
         patch.object(evaluate_all, "evaluate_ragas", return_value={"retrieval_only": {"mrr": 0.9}, "full": None}):
        results = evaluate_all.run_all_evaluations()

    assert "ragas" in results
    assert results["ragas"] == {"retrieval_only": {"mrr": 0.9}, "full": None}

    report_path = tmp_path / "fine_tuning" / "data" / "evaluation_report.json"
    assert report_path.exists()
    with open(report_path) as f:
        saved = json.load(f)
    assert saved["ragas"] == {"retrieval_only": {"mrr": 0.9}, "full": None}


def test_evaluate_ragas_never_calls_retrieval_or_llm(tmp_path):
    retrieval_path = _write(tmp_path, "ragas_scores_retrieval_only.json", RETRIEVAL_ONLY_FIXTURE)
    full_path = _write(tmp_path, "ragas_scores_full.json", FULL_FIXTURE)

    def boom(*args, **kwargs):
        raise AssertionError("should not be called")

    patches = []
    try:
        from src.rag import retriever as _retriever_mod
        patches.append(patch.object(_retriever_mod, "retrieve_and_rerank", side_effect=boom))
    except ImportError:
        pass
    try:
        from src.utils import openai_utils as _openai_mod
        patches.append(patch.object(_openai_mod, "call_openai_structured", side_effect=boom))
    except ImportError:
        pass
    try:
        import ragas as _ragas_mod
        patches.append(patch.object(_ragas_mod, "evaluate", side_effect=boom))
    except ImportError:
        pass

    for p in patches:
        p.start()
    try:
        with patch("fine_tuning.evaluate_all.Path") as MockPath:
            def side_effect(arg):
                if "retrieval_only" in arg:
                    return retrieval_path
                if "full" in arg:
                    return full_path
                return Path(arg)
            MockPath.side_effect = side_effect
            result = evaluate_all.evaluate_ragas()
        assert result is not None
    finally:
        for p in patches:
            p.stop()
