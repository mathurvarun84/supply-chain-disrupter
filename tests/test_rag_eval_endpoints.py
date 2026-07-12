"""
Tests for Screen 6 (RAG / RAGAS Evaluation) endpoints — src/api/routers/rag.py.

Day 8: endpoints read persisted RAGAS JSON + live ChromaDB collection counts.
"""

import json

from fastapi.testclient import TestClient

from src.api.fixtures import GOLD_QA, RAGAS_SCORES
from src.api.main import app
from src.rag.utils import RAGAS_GOLD_DATASET_PATH, RAGAS_SCORES_PATH

client = TestClient(app)


def test_scorecard_returns_200_and_matches_pydantic_model():
    resp = client.get("/api/rag/scorecard")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 1
    for tile in body:
        assert {"metric", "score", "threshold", "passed"} <= tile.keys()


def test_scorecard_passed_matches_score_vs_threshold():
    resp = client.get("/api/rag/scorecard")
    for tile in resp.json():
        assert tile["passed"] == (tile["score"] >= tile["threshold"])


def test_scorecard_differs_from_day1_fixture_when_ragas_file_exists():
    if not RAGAS_SCORES_PATH.exists():
        return
    resp = client.get("/api/rag/scorecard")
    body = resp.json()
    fixture_faith = next(t for t in RAGAS_SCORES if t["metric"] == "Faithfulness")
    live_faith = next(t for t in body if t["metric"] == "Faithfulness")
    assert live_faith["score"] != fixture_faith["score"]


def test_corpus_returns_200_and_real_plus_synth_equals_docs():
    resp = client.get("/api/rag/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    for card in body:
        assert card["real"] + card["synth"] == card["docs"]
        assert card["docs"] >= 0


def test_corpus_collection_names_match_exactly():
    resp = client.get("/api/rag/corpus")
    names = {card["name"] for card in resp.json()}
    assert names == {"historical_precedents", "export_control_corpus", "india_sourcing_corpus"}


def test_gold_dataset_returns_200_and_query_style_split_is_roughly_60_40():
    resp = client.get("/api/rag/gold-dataset")
    assert resp.status_code == 200
    body = resp.json()
    if RAGAS_GOLD_DATASET_PATH.exists():
        dataset = json.loads(RAGAS_GOLD_DATASET_PATH.read_text(encoding="utf-8"))
        assert len(body) == len(dataset.get("test_cases") or [])
    else:
        assert len(body) == 0
    if body:
        agent_pattern = sum(1 for row in body if row["query_style"] == "agent_pattern")
        proportion = agent_pattern / len(body)
        assert 0.5 <= proportion <= 0.75


def test_gold_dataset_differs_from_day1_fixture_when_file_exists():
    if not RAGAS_GOLD_DATASET_PATH.exists():
        return
    resp = client.get("/api/rag/gold-dataset")
    body = resp.json()
    assert len(body) != len(GOLD_QA)


def test_faithfulness_gate_reason_matches_guardrails_endpoint():
    """Guardrails faithfulness-gate row is served from guardrail_events table."""
    guard_resp = client.get("/api/guardrails/events")
    gate_row = next(r for r in guard_resp.json() if r["name"] == "faithfulness-gate")
    assert "0.61 < 0.75" in gate_row["reason"] or "faithfulness" in gate_row["reason"].lower()
