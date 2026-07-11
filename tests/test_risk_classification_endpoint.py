"""
Tests for Screen 2 (Risk Classification) endpoints — src/api/routers/risk.py.

Mocks run_distilbert_inference / run_llm_signal / run_judge so no real
model weights or OPENAI_API_KEY are needed — no network/model calls
happen in this test module. Seeds a temp SQLite DB via monkeypatching
src.utils.db_utils.DB_PATH, same pattern as test_live_feed_endpoints.py.
"""

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.agents.state import DistilBERTSignal, JudgeVerdict, LLMSignal
from src.api.main import app

client = TestClient(app)


def _seed_daily_records(conn, order_id=101, order_date="2024-03-01"):
    conn.execute(
        """
        CREATE TABLE daily_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            event_date TEXT,
            port TEXT,
            sku TEXT,
            order_region TEXT,
            delivery_status TEXT,
            natural_disaster_risk REAL,
            supply_disruption_index REAL,
            defect_rate_pct REAL,
            export_control_level REAL,
            risk_score_composite REAL,
            disruption_event_label TEXT,
            year INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO daily_records (order_id, event_date, port, sku, order_region, "
        "delivery_status, natural_disaster_risk, supply_disruption_index, defect_rate_pct, "
        "export_control_level, risk_score_composite, disruption_event_label, year) VALUES "
        "(?, ?, 'Hsinchu', 'SKU-1', 'Eastern Asia', 'Late delivery', 5.0, 6.0, 10.0, 2.0, 0.6, 'HIGH', 2024)",
        (order_id, order_date),
    )


def _seed_lite_master(conn):
    """_get_norm_bounds() (lru_cache'd in risk_classifier_agent) reads
    MIN/MAX bounds from lite_master — seed a minimal table so it doesn't
    blow up on a missing-table error in the test DB."""
    conn.execute(
        "CREATE TABLE lite_master (weather_severity_hub REAL, natural_disaster_risk REAL, "
        "supply_disruption_index REAL, defect_rate_pct REAL, disruption_news_count REAL)"
    )
    conn.execute(
        "INSERT INTO lite_master VALUES (1.18, 1.18, 4.09, 2.0, 0.0), (10.0, 10.0, 9.97, 19.82, 5.0)"
    )


def _seed_risk_classifications_table(conn):
    conn.execute(
        """
        CREATE TABLE risk_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            mode TEXT NOT NULL,
            composite_score REAL NOT NULL,
            geo_component REAL,
            supply_component REAL,
            freight_component REAL,
            defect_component REAL,
            duration_days REAL,
            base_label TEXT,
            final_label TEXT,
            escalated INTEGER,
            rag_citations TEXT,
            rationale TEXT,
            run_ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


@pytest.fixture
def db_with_record(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    _seed_daily_records(conn)
    _seed_risk_classifications_table(conn)
    _seed_lite_master(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

    from src.agents.risk_classifier_agent.agent import _get_norm_bounds
    _get_norm_bounds.cache_clear()

    return db_path


@pytest.fixture(autouse=True)
def _no_real_chroma():
    """HIGH/CRITICAL labels trigger a ChromaDB RAG citation lookup inside
    risk_classifier_agent — stub it so tests don't depend on a populated
    outputs/chromadb corpus."""
    with patch("src.agents.risk_classifier_agent.agent.query_chroma_rag", return_value=[]):
        yield


def _mock_ensemble():
    """Patch the three non-blocking ensemble calls used inside
    risk_classifier_agent so tests never hit a model file or OpenAI."""
    return (
        patch(
            "src.agents.risk_classifier_agent.agent.run_distilbert_inference",
            return_value=DistilBERTSignal(
                predicted_label="HIGH",
                confidence=0.8,
                probability_distribution={"LOW": 0.1, "MEDIUM": 0.1, "HIGH": 0.8, "CRITICAL": 0.0},
                model_source="fine-tuned",
                inference_ms=12.0,
            ),
        ),
        patch(
            "src.agents.risk_classifier_agent.agent.run_llm_signal",
            return_value=LLMSignal(
                predicted_label="HIGH",
                rationale="Late delivery signal dominates.",
                rag_citations=["Some_Source"],
                rag_chunks_used=2,
                confidence_level="high",
                primary_driver="delivery_status",
            ),
        ),
        patch(
            "src.agents.risk_classifier_agent.agent.run_judge",
            return_value=JudgeVerdict(
                final_label="HIGH",
                verdict_type="unanimous",
                reasoning="All three signals agree on HIGH.",
                signals_agreed=True,
                disagreement_explanation=None,
                final_critical_flag=False,
            ),
        ),
    )


def test_uncached_order_runs_ensemble_once(db_with_record):
    p1, p2, p3 = _mock_ensemble()
    with p1 as m_db, p2 as m_llm, p3 as m_judge:
        resp = client.get("/api/risk-classification/101")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "101"
        assert body["from_cache"] is False
        assert body["final_label"] == "HIGH"
        assert body["distilbert_signal"]["predicted_label"] == "HIGH"
        assert body["llm_signal"]["predicted_label"] == "HIGH"
        assert body["judge_verdict"]["verdict_type"] == "unanimous"
        assert m_db.call_count == 1
        assert m_llm.call_count == 1
        assert m_judge.call_count == 1


def test_cached_row_returned_without_recomputing(db_with_record):
    conn = sqlite3.connect(db_with_record)
    conn.execute(
        "INSERT INTO risk_classifications (order_id, mode, composite_score, geo_component, "
        "supply_component, freight_component, defect_component, duration_days, base_label, "
        "final_label, escalated, rag_citations, rationale) VALUES "
        "(101, 'replay', 0.6, 0.5, 0.6, 0.4, 0.3, NULL, 'HIGH', 'HIGH', 0, '[]', 'cached rationale')"
    )
    conn.commit()
    conn.close()

    p1, p2, p3 = _mock_ensemble()
    with p1 as m_db, p2 as m_llm, p3 as m_judge:
        resp = client.get("/api/risk-classification/101")
        assert resp.status_code == 200
        body = resp.json()
        assert body["from_cache"] is True
        assert body["final_label"] == "HIGH"
        assert m_db.call_count == 0
        assert m_llm.call_count == 0
        assert m_judge.call_count == 0


def test_final_critical_flag_always_server_derived(db_with_record):
    """Even if the mocked Judge disagrees with its own final_critical_flag
    field, the endpoint must recompute final_critical_flag/slack_should_fire
    strictly as (final_label == 'CRITICAL') rather than trusting it."""
    judge_says_critical_but_flag_false = JudgeVerdict(
        final_label="CRITICAL",
        verdict_type="unanimous",
        reasoning="Shipping canceled hard rule.",
        signals_agreed=True,
        disagreement_explanation=None,
        final_critical_flag=False,  # deliberately wrong/untrusted value
    )
    p1, p2, _ = _mock_ensemble()
    with p1, p2, patch(
        "src.agents.risk_classifier_agent.agent.run_judge",
        return_value=judge_says_critical_but_flag_false,
    ):
        resp = client.get("/api/risk-classification/101")
        body = resp.json()
        assert body["final_label"] == "CRITICAL"
        assert body["final_critical_flag"] is True
        assert body["slack_should_fire"] is True


def test_partial_signals_never_500(db_with_record):
    """DistilBERT/LLM/Judge all unavailable — endpoint must still return
    200 with final_label falling back to rule_signal.escalated_label."""
    with patch(
        "src.agents.risk_classifier_agent.agent.run_distilbert_inference",
        return_value=DistilBERTSignal(
            predicted_label="N/A",
            confidence=0.0,
            probability_distribution={},
            model_source="not-available-skipped",
            inference_ms=0.0,
        ),
    ), patch(
        "src.agents.risk_classifier_agent.agent.run_llm_signal", return_value=None
    ), patch(
        "src.agents.risk_classifier_agent.agent.run_judge", return_value=None
    ):
        resp = client.get("/api/risk-classification/101")
        assert resp.status_code == 200
        body = resp.json()
        assert body["distilbert_signal"]["predicted_label"] == "N/A"
        assert body["llm_signal"]["predicted_label"] is None
        assert body["judge_verdict"] is None
        assert body["final_label"] == body["rule_signal"]["escalated_label"]


def test_unknown_order_id_returns_404(db_with_record):
    resp = client.get("/api/risk-classification/999999")
    assert resp.status_code == 404


def test_invalid_run_id_returns_404(db_with_record):
    resp = client.get("/api/risk-classification/not-a-number")
    assert resp.status_code == 404


def test_latest_resolves_to_most_recent_order(db_with_record):
    p1, p2, p3 = _mock_ensemble()
    with p1, p2, p3:
        resp = client.get("/api/risk-classification/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["order_id"] == 101
