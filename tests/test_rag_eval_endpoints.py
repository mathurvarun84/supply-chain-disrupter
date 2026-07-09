"""
Tests for Screen 6 (RAG / RAGAS Evaluation) endpoints — src/api/routers/rag.py.

Pure fixture endpoints (RAGAS_SCORES / CORPUS / GOLD_QA in
src/api/fixtures.py) — no ChromaDB, RAGAS run, or network calls happen in
this test module, matching the "stub against fixture JSON" cut of Day 7.
"""

from fastapi.testclient import TestClient

from src.api.fixtures import CORPUS, GOLD_QA, RAGAS_SCORES
from src.api.main import app

client = TestClient(app)


def test_scorecard_returns_200_and_matches_pydantic_model():
    resp = client.get("/api/rag/scorecard")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == len(RAGAS_SCORES)
    for tile in body:
        assert {"metric", "score", "threshold", "passed"} <= tile.keys()


def test_scorecard_passed_matches_score_vs_threshold():
    resp = client.get("/api/rag/scorecard")
    for tile in resp.json():
        assert tile["passed"] == (tile["score"] >= tile["threshold"])


def test_corpus_returns_200_and_real_plus_synth_equals_docs():
    resp = client.get("/api/rag/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == len(CORPUS)
    for card in body:
        assert card["real"] + card["synth"] == card["docs"]


def test_corpus_collection_names_match_exactly():
    resp = client.get("/api/rag/corpus")
    names = {card["name"] for card in resp.json()}
    assert names == {"historical_precedents", "export_control_corpus", "india_sourcing_corpus"}


def test_gold_dataset_returns_200_and_query_style_split_is_roughly_60_40():
    resp = client.get("/api/rag/gold-dataset")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == len(GOLD_QA)
    agent_pattern = sum(1 for row in body if row["query_style"] == "agent_pattern")
    proportion = agent_pattern / len(body)
    assert 0.5 <= proportion <= 0.75


def test_faithfulness_example_pair_matches_guardrails_fixture():
    """RAGAS_SCORES' faithfulness score (0.87, passing at threshold 0.75)
    and GUARDRAIL_TABLE's faithfulness-gate failure reason (0.61 < 0.75)
    are the same pair the Faithfulness Gate panel renders — they must not
    silently diverge from each other."""
    from src.api.fixtures import GUARDRAIL_TABLE

    faithfulness_tile = next(t for t in RAGAS_SCORES if t["metric"] == "Faithfulness")
    assert faithfulness_tile["score"] == 0.87
    assert faithfulness_tile["threshold"] == 0.75

    gate_row = next(r for r in GUARDRAIL_TABLE if r["name"] == "faithfulness-gate")
    assert "0.61 < 0.75" in gate_row["reason"]
